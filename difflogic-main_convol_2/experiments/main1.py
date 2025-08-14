import argparse
import math
import random
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

# Set environment for CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import gc

torch.set_num_threads(1)
BITS_TO_TORCH_FLOATING_POINT_TYPE = {16: torch.float16, 32: torch.float32, 64: torch.float64}

# --- Binary Operation Functions ---
def bin_op_s(a, b, i_s):
    ops = torch.stack([
        torch.zeros_like(a),
        a * b,
        a * (1 - b),
        a,
        (1 - a) * b,
        b,
        a + b - 2 * a * b,
        a + b - a * b,
        1 - (a + b - a * b),
        1 - (a + b - 2 * a * b),
        1 - b,
        1 - b + a * b,
        1 - a,
        1 - a + a * b,
        1 - a * b,
        torch.ones_like(a)
    ], dim=-1)
    return torch.einsum('...i,...i->...', ops, i_s)

# --- Gradient Scaling ---
class GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.factor, None

# --- Unique Connections Helper ---
def get_unique_connections(in_dim, out_dim, device='cuda'):
    assert out_dim * 2 >= in_dim, ('The number of neurons ({}) must not be smaller than half of the number of inputs '
                                   '({}) because otherwise not all inputs could be used or considered.'.format(out_dim, in_dim))
    x = torch.arange(in_dim).long().unsqueeze(0)
    a, b = x[..., ::2], x[..., 1::2]
    if a.shape[-1] != b.shape[-1]:
        m = min(a.shape[-1], b.shape[-1])
        a = a[..., :m]
        b = b[..., :m]
    if a.shape[-1] < out_dim:
        a_, b_ = x[..., 1::2], x[..., 2::2]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        if a.shape[-1] != b.shape[-1]:
            m = min(a.shape[-1], b.shape[-1])
            a = a[..., :m]
            b = b[..., :m]
    offset = 2
    while out_dim > a.shape[-1] > offset:
        a_, b_ = x[..., :-offset], x[..., offset:]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        offset += 1
        assert a.shape[-1] == b.shape[-1], (a.shape[-1], b.shape[-1])
    if a.shape[-1] >= out_dim:
        a = a[..., :out_dim]
        b = b[..., :out_dim]
    else:
        assert False, (a.shape[-1], offset, out_dim)
    perm = torch.randperm(out_dim)
    a = a[:, perm].squeeze(0)
    b = b[:, perm].squeeze(0)
    a, b = a.to(torch.int64), b.to(torch.int64)
    a, b = a.to(device), b.to(device)
    return a, b

# --- Convolutional Logic Tree Layer ---
class ConvLogicTreeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 tree_depth=3, grad_factor=1.0, device='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grad_factor = grad_factor
        self.device = device
        self.tree_depth = tree_depth

        self.patch_size = kernel_size * kernel_size
        self.num_leaves = 2 ** tree_depth
        assert self.patch_size * 2 >= self.num_leaves, "Receptive field is too small for 2 channels."

        # Restrict to 2 input channels per tree
        self.leaf_indices = []
        for _ in range(out_channels):
            channel_indices = torch.randperm(in_channels)[:2]
            for c in channel_indices:
                spatial_indices = torch.randperm(self.patch_size)[:self.num_leaves // 2]
                leaf_idx = c * self.patch_size + spatial_indices
                self.leaf_indices.append(leaf_idx)
            self.leaf_indices[-1] = torch.cat(self.leaf_indices[-2:], dim=0)
            self.leaf_indices.pop(-2)
        self.leaf_indices = torch.stack(self.leaf_indices, dim=0).to(device)

        num_gates = self.num_leaves - 1
        self.weights = nn.Parameter(torch.randn(out_channels, num_gates, 16, device=device))
        with torch.no_grad():
            self.weights.fill_(0.0)
            self.weights[:, :, 3] = 5.0

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        batch_size, _, H, W = x.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        L = H_out * W_out

        patches = self.unfold(x)
        patches = patches.transpose(1, 2)

        if self.grad_factor != 1.0:
            patches = GradFactor.apply(patches, self.grad_factor)

        outputs = []
        chunk_size = 256
        for c in range(self.out_channels):
            leaves = torch.gather(
                patches,
                dim=2,
                index=self.leaf_indices[c].view(1, 1, -1).expand(batch_size, L, -1)
            )
            weights = self.weights[c]
            current = leaves
            for level in range(self.tree_depth):
                new_current = []
                for i in range(0, current.shape[-1], 2):
                    result = torch.zeros(batch_size, L, device=self.device)
                    for start in range(0, L, chunk_size):
                        end = min(start + chunk_size, L)
                        a = current[:, start:end, i]
                        b = current[:, start:end, i+1]
                        gate_idx = (2**level - 1) + (i//2)
                        w = weights[gate_idx]
                        if self.training:
                            w = F.softmax(w, dim=-1)
                        else:
                            w = torch.zeros_like(w)
                            w[w.argmax()] = 1.0
                        chunk_result = bin_op_s(a, b, w)
                        result[:, start:end] = chunk_result
                    new_current.append(result)
                current = torch.stack(new_current, dim=-1)
            outputs.append(current.squeeze(-1))

        out = torch.stack(outputs, dim=1)
        return out.view(batch_size, self.out_channels, H_out, W_out)

# --- Or Pool Layer ---
class OrPoolLayer(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

# --- Differentiable Logic Layer ---
class DiffLogicLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, device: str = 'cuda', grad_factor: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor

        self.conn_a, self.conn_b = get_unique_connections(in_dim, out_dim, device)
        self.weights = nn.Parameter(torch.randn(out_dim, 16, device=device))
        with torch.no_grad():
            self.weights.fill_(0.0)
            self.weights[:, 3] = 5.0

    def forward(self, x):
        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)
        batch_size = x.shape[0]
        a = x[:, self.conn_a]
        b = x[:, self.conn_b]
        weights = F.softmax(self.weights, dim=-1) if self.training else torch.zeros_like(self.weights).scatter_(-1, self.weights.argmax(-1, keepdim=True), 1.0)
        out = bin_op_s(a, b, weights)
        return out

# --- Group Sum ---
class GroupSum(nn.Module):
    def __init__(self, k: int, tau: float = 1.0, device: str = 'cuda'):
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device
    def forward(self, x):
        new_shape = x.shape[:-1] + (self.k, x.shape[-1] // self.k)
        return x.view(new_shape).sum(-1) / self.tau

# --- CIFAR-10 ConvLogic Network ---
class ConvLogicNetCIFAR(nn.Module):
    def __init__(self, num_classes: int, k: int = 32, tau: float = 10.0, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.tau = tau
        self.conv_block1 = nn.Sequential(
            ConvLogicTreeLayer(in_channels=9, out_channels=k, kernel_size=3, stride=1, padding=1, tree_depth=3, device=device),
            OrPoolLayer(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            ConvLogicTreeLayer(in_channels=k, out_channels=4*k, kernel_size=3, stride=1, padding=1, tree_depth=3, device=device),
            OrPoolLayer(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            ConvLogicTreeLayer(in_channels=4*k, out_channels=16*k, kernel_size=3, stride=1, padding=1, tree_depth=3, device=device),
            OrPoolLayer(kernel_size=2, stride=2)
        )
        self.conv_block4 = nn.Sequential(
            ConvLogicTreeLayer(in_channels=16*k, out_channels=32*k, kernel_size=3, stride=1, padding=1, tree_depth=3, device=device),
            OrPoolLayer(kernel_size=2, stride=2)
        )
        self.conv_blocks = nn.ModuleList([
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4
        ])
        self.flatten = nn.Flatten()
        fc_in_dim = 32 * k * 2 * 2
        self.fc1 = DiffLogicLayer(in_dim=fc_in_dim, out_dim=1280 * k, device=device)
        self.fc2 = DiffLogicLayer(in_dim=1280 * k, out_dim=640 * k, device=device)
        self.fc3 = DiffLogicLayer(in_dim=640 * k, out_dim=320 * k, device=device)
        self.classifier = GroupSum(k=num_classes, tau=self.tau, device=device)

    def forward(self, x):
        # Preprocess inside forward for GPU acceleration
        x = torch.cat([(x > ((i+1)/4)).float() for i in range(3)], dim=1)
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        # Disable checkpointing to avoid warning and test performance
        for block in self.conv_blocks:
            x = block(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.classifier(x)
        return x

# --- Memory Logging ---
def log_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")
        return allocated, cached
    return 0, 0

# --- Data Loading Functions ---
def load_dataset(args):
    validation_loader = None
    if args.dataset == 'cifar-10-3-thresholds':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_set = torchvision.datasets.CIFAR10('./data-cifar', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10('./data-cifar', train=False, transform=transform)
        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=2)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size,
                                                        shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                  shuffle=False, pin_memory=True, drop_last=True)
    else:
        raise NotImplementedError(f'The dataset {args.dataset} is not supported!')
    return train_loader, validation_loader, test_loader

def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break

def num_classes_of_dataset(dataset):
    return 10

# --- Model Selection ---
def get_model(args):
    class_count = num_classes_of_dataset(args.dataset)
    tau_values = {32: 20, 256: 40, 512: 280, 1024: 340, 2048: 450}
    tau = tau_values.get(args.num_neurons, 40)
    model = ConvLogicNetCIFAR(num_classes=class_count, k=args.num_neurons, tau=tau, device='cuda')
    model = model.to('cuda')
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, loss_fn, optimizer

# --- Training and Evaluation ---
def train(model, x, y, loss_fn, optimizer):
    start_time = time.time()
    outputs = model(x)
    loss = loss_fn(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), time.time() - start_time

def eval_model(model, loader, mode, max_batches=5):
    start_time = time.time()
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        accs = []
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to('cuda'), y.to('cuda')
            outputs = model(x)
            preds = outputs.argmax(dim=-1)
            accs.append((preds == y).float().mean().item())
        model.train(mode=orig_mode)
    return np.mean(accs) if accs else 0.0, time.time() - start_time

# --- Main Script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train convolutional differentiable logic gate network.')
    parser.add_argument('--dataset', type=str, choices=['cifar-10-3-thresholds'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.02)
    parser.add_argument('--training-bit-count', type=int, default=32)
    parser.add_argument('--num-iterations', type=int, default=4500)
    parser.add_argument('--eval-freq', type=int, default=2)
    parser.add_argument('--valid-set-size', type=float, default=0.1)
    parser.add_argument('--extensive-eval', action='store_true')
    parser.add_argument('--num-neurons', type=int, required=True)
    parser.add_argument('--grad-factor', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=0.002)
    args = parser.parse_args()

    print(vars(args))
    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)
    model, loss_fn, optim = get_model(args)
    best_acc = 0

    try:
        for i, (x, y) in tqdm(enumerate(load_n(train_loader, args.num_iterations)), desc='iteration', total=args.num_iterations):
            start_iter_time = time.time()
            print(f"\nIteration {i+1}:")
            log_memory()
            x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
            y = y.to('cuda')
            loss_val, train_time = train(model, x, y, loss_fn, optim)
            print(f"Training time: {train_time:.2f}s, Loss: {loss_val:.4f}")

            if (i+1) % args.eval_freq == 0:
                torch.cuda.empty_cache()
                gc.collect()
                log_memory()
                train_acc_eval_mode, train_eval_time = eval_model(model, train_loader, mode=False, max_batches=5)
                test_acc_eval_mode, test_eval_time = eval_model(model, test_loader, mode=False, max_batches=5)
                valid_acc_eval_mode, valid_eval_time = -1, 0
                if args.extensive_eval and validation_loader is not None:
                    valid_acc_eval_mode, valid_eval_time = eval_model(model, validation_loader, mode=False, max_batches=5)
                r = {
                    'train_acc_eval_mode': train_acc_eval_mode,
                    'valid_acc_eval_mode': valid_acc_eval_mode,
                    'test_acc_eval_mode': test_acc_eval_mode,
                    'loss': loss_val,
                }
                print(f"Iter {i+1}: {r}")
                print(f"Evaluation times: Train={train_eval_time:.2f}s, Valid={valid_eval_time:.2f}s, Test={test_eval_time:.2f}s")
                if valid_acc_eval_mode > best_acc:
                    best_acc = valid_acc_eval_mode
                    print("New best validation accuracy!")
                    torch.save(model.state_dict(), f"best_model_{args.dataset}_k{args.num_neurons}.pt")

            iter_time = time.time() - start_iter_time
            print(f"Total iteration time: {iter_time:.2f}s")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        final_model_path = f"model_interrupted_{args.dataset}_k{args.num_neurons}.pt"
        torch.save(model.state_dict(), final_model_path)
        print(f"Model saved to {final_model_path}")
        exit(0)

    final_model_path = f"model_final_{args.dataset}_k{args.num_neurons}.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")