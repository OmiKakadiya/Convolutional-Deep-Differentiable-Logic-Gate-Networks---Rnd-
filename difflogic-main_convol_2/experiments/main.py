import argparse
import math
import random
import os
import time  # for timing

import numpy as np
import torch.nn as nn
import torch
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import gc
torch.cuda.empty_cache()
gc.collect()

from torch.utils.checkpoint import checkpoint_sequential

from results_json import ResultsJSON

import mnist_dataset
import uci_datasets
from difflogic import LogicLayer, GroupSum, PackBitsTensor, CompiledLogicNet

import torch
import numpy as np
# -----------------------------------------------------------------------------------
# --- Integrated Convolutional Differentiable Logic Network Code -------------------
# -----------------------------------------------------------------------------------

def bin_op(a, b, i):
    # Vectorized implementation of binary operations
    if i == 0:      # constant 0
        return torch.zeros_like(a)
    elif i == 1:    # A and B
        return a * b
    elif i == 2:    # not(A implies B)
        return a * (1 - b)
    elif i == 3:    # A
        return a
    elif i == 4:    # not(B implies A)
        return (1 - a) * b
    elif i == 5:    # B
        return b
    elif i == 6:    # A xor B
        return a + b - 2 * a * b
    elif i == 7:    # A or B
        return a + b - a * b
    elif i == 8:    # not(A or B)
        return 1 - (a + b - a * b)
    elif i == 9:    # not(A xor B)
        return 1 - (a + b - 2 * a * b)
    elif i == 10:   # not(B)
        return 1 - b
    elif i == 11:   # B implies A
        return 1 - b + a * b
    elif i == 12:   # not(A)
        return 1 - a
    elif i == 13:   # A implies B
        return 1 - a + a * b
    elif i == 14:   # not(A and B)
        return 1 - a * b
    elif i == 15:   # constant 1
        return torch.ones_like(a)
    else:
        raise ValueError("Operator index must be in 0...15")

def bin_op_s(a, b, i_s):
    # Computes a weighted sum of the 16 binary operations (vectorized)
    ops = torch.stack([
        torch.zeros_like(a),                    # 0: constant 0
        a * b,                                  # 1: A and B
        a * (1 - b),                            # 2: not(A implies B)
        a,                                      # 3: A
        (1 - a) * b,                            # 4: not(B implies A)
        b,                                      # 5: B
        a + b - 2 * a * b,                      # 6: A xor B
        a + b - a * b,                          # 7: A or B
        1 - (a + b - a * b),                    # 8: not(A or B)
        1 - (a + b - 2 * a * b),                # 9: not(A xor B)
        1 - b,                                  # 10: not(B)
        1 - b + a * b,                          # 11: B implies A
        1 - a,                                  # 12: not(A)
        1 - a + a * b,                          # 13: A implies B
        1 - a * b,                              # 14: not(A and B)
        torch.ones_like(a)                      # 15: constant 1
    ], dim=-1)  # Shape: (batch, L, 16)
    
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

class ConvLogicTreeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 tree_depth=3, connections='random', grad_factor=1.0, device='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grad_factor = grad_factor
        self.device = device
        self.tree_depth = tree_depth

        self.patch_size = in_channels * kernel_size * kernel_size
        self.num_leaves = 2 ** tree_depth
        assert self.patch_size >= self.num_leaves, "Receptive field is too small."

        if connections == 'random':
            self.leaf_indices = torch.stack([
                torch.randint(0, self.patch_size, (self.num_leaves,)) for _ in range(out_channels)
            ], dim=0).to(device)
        elif connections == 'unique':
            leaf_list = []
            for _ in range(out_channels):
                perm = torch.randperm(self.patch_size)[:self.num_leaves]
                leaf_list.append(perm)
            self.leaf_indices = torch.stack(leaf_list, dim=0).to(device)
        else:
            raise ValueError(connections)

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
        chunk_size = 256  # Fixed chunk size for memory efficiency
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

# --- Fully Connected Logic Layer ---
class LogicLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, device: str = 'cuda', grad_factor: float = 1.0, connections: str = 'unique'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.linear = nn.Linear(in_dim, out_dim).to(device)
    def forward(self, x):
        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)
        return self.linear(x)

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
    def __init__(self,
                 num_classes: int,
                 k: int = 32,
                 tau: float = 10.0,
                 device: str = 'cuda'):
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
        self.fc1 = LogicLayer(in_dim=fc_in_dim, out_dim=1280 * k, device=device)
        self.fc2 = LogicLayer(in_dim=1280 * k, out_dim=640 * k, device=device)
        self.fc3 = LogicLayer(in_dim=640 * k, out_dim=320 * k, device=device)
        self.classifier = GroupSum(k=num_classes, tau=self.tau, device=device)

    def forward(self, x):
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        if self.training:
            x = checkpoint_sequential(self.conv_blocks, segments=4, input=x, use_reentrant=False)
        else:
            for block in self.conv_blocks:
                x = block(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.classifier(x)
        return x

# --- MNIST ConvLogic Network ---
class ConvLogicNetMNIST(nn.Module):
    def __init__(self,
                 num_classes: int,
                 k: int = 32,
                 tau: float = 10.0,
                 device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.tau = tau
        self.conv_block1 = nn.Sequential(
            ConvLogicTreeLayer(in_channels=1, out_channels=k, kernel_size=5, stride=1, padding=0, tree_depth=3, device=device),
            OrPoolLayer(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            ConvLogicTreeLayer(in_channels=k, out_channels=3*k, kernel_size=3, stride=1, padding=1, tree_depth=3, device=device),
            OrPoolLayer(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            ConvLogicTreeLayer(in_channels=3*k, out_channels=9*k, kernel_size=3, stride=1, padding=1, tree_depth=3, device=device),
            OrPoolLayer(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = LogicLayer(in_dim=9*k*3*3, out_dim=k, device=device)
        self.fc2 = LogicLayer(in_dim=k, out_dim=k, device=device)
        self.classifier = GroupSum(k=num_classes, tau=self.tau, device=device)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        return x

# -----------------------------------------------------------------------------------
# --- End Integrated Conv Diff Logic Code -----------------------------------------
# -----------------------------------------------------------------------------------

torch.set_num_threads(1)
BITS_TO_TORCH_FLOATING_POINT_TYPE = {16: torch.float16, 32: torch.float32, 64: torch.float64}

# --- Memory Logging ---
def log_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# --- Data Loading Functions ---
def load_dataset(args):
    validation_loader = None
    if args.dataset == 'adult':
        train_set = uci_datasets.AdultDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.AdultDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset == 'breast_cancer':
        train_set = uci_datasets.BreastCancerDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.BreastCancerDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset.startswith('monk'):
        style = int(args.dataset[4])
        train_set = uci_datasets.MONKsDataset('./data-uci', style, split='train', download=True, with_val=False)
        test_set = uci_datasets.MONKsDataset('./data-uci', style, split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    elif args.dataset in ['mnist', 'mnist20x20']:
        remove_border = (args.dataset=='mnist20x20')
        train_set = mnist_dataset.MNIST('./data-mnist', train=True, download=True, remove_border=remove_border)
        test_set = mnist_dataset.MNIST('./data-mnist', train=False, remove_border=remove_border)
        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size,
                                                        shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                  shuffle=False, pin_memory=True, drop_last=True)
    elif 'cifar-10' in args.dataset:
        transform = {
            'cifar-10-3-thresholds': lambda x: torch.cat([(x > ((i+1)/4)).float() for i in range(3)], dim=0),
            'cifar-10-31-thresholds': lambda x: torch.cat([(x > ((i+1)/32)).float() for i in range(31)], dim=0),
        }[args.dataset]
        tfms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(transform),
        ])

        train_set = torchvision.datasets.CIFAR10('./data-cifar', train=True, download=True, transform=tfms)
        test_set = torchvision.datasets.CIFAR10('./data-cifar', train=False, transform=tfms)
        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
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

def input_dim_of_dataset(dataset):
    dims = {
        'adult': 116,
        'breast_cancer': 51,
        'monk1': 17,
        'monk2': 17,
        'monk3': 17,
        'mnist': 784,
        'mnist20x20': 400,
        'cifar-10-3-thresholds': 3 * 32 * 32 * 3,
        'cifar-10-31-thresholds': 3 * 32 * 32 * 31,
    }
    return dims[dataset]

def num_classes_of_dataset(dataset):
    classes = {
        'adult': 2,
        'breast_cancer': 2,
        'monk1': 2,
        'monk2': 2,
        'monk3': 2,
        'mnist': 10,
        'mnist20x20': 10,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }
    return classes[dataset]

# --- Model Selection ---
def get_model(args):
    class_count = num_classes_of_dataset(args.dataset)
    if args.architecture == 'conv_logic':
        if args.dataset in ['mnist', 'mnist20x20']:
            model = ConvLogicNetMNIST(num_classes=class_count, k=args.num_neurons, tau=args.tau, device='cuda')
        elif 'cifar' in args.dataset:
            model = ConvLogicNetCIFAR(num_classes=class_count, k=args.num_neurons, tau=args.tau, device='cuda')
        else:
            raise NotImplementedError("Conv logic architecture is not implemented for this dataset")
    elif args.architecture == 'randomly_connected':
        raise NotImplementedError("Randomly connected architecture not implemented in this integrated conv version")
    else:
        raise NotImplementedError(args.architecture)
    model = model.to('cuda')
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, loss_fn, optimizer

def train(model, x, y, loss_fn, optimizer):
    outputs = model(x)
    loss = loss_fn(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_model(model, loader, mode, max_batches=10):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        accs = []
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            outputs = model(x.to('cuda').round())
            preds = outputs.argmax(dim=-1)
            accs.append((preds == y.to('cuda')).float().mean().item())
        model.train(mode=orig_mode)
    return np.mean(accs) if accs else 0.0

def packbits_eval(model, loader):
    orig_mode = model.training
    with torch.no_grad():
        model.eval()
        res = np.mean(
            [
                (model(PackBitsTensor(x.to('cuda').reshape(x.shape[0], -1).round().bool())).argmax(-1) == y.to(
                    'cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()

# --- Main Script: Argument Parsing and Training Loop ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train convolutional differentiable logic gate network.')
    parser.add_argument('-eid', '--experiment_id', type=int, default=None)
    parser.add_argument('--dataset', type=str, choices=[
        'adult', 'breast_cancer', 'monk1', 'monk2', 'monk3',
        'mnist', 'mnist20x20', 'cifar-10-3-thresholds', 'cifar-10-31-thresholds'
    ], required=True, help='the dataset to use')
    parser.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature tau')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=16, help='batch size (default: 16)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')
    parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python'],
                        help='`cuda` is fast; `python` is simpler but slower')
    parser.add_argument('--packbits_eval', action='store_true', help='Use PackBitsTensor for additional evaluation.')
    parser.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')
    parser.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=2000, help='Evaluation frequency (default: 2000)')
    parser.add_argument('--valid-set-size', '-vss', type=float, default=0.0, help='Fraction of train set used for validation (default: 0)')
    parser.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval)')
    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
    parser.add_argument('--architecture', '-a', type=str, default='conv_logic', choices=['conv_logic', 'randomly_connected'])
    parser.add_argument('--num_neurons', '-k', type=int, required=True, help='Base width (k) for the network')
    parser.add_argument('--num_layers', '-l', type=int, default=4, help='Number of layers (for randomly_connected; not used in conv_logic)')
    parser.add_argument('--grad-factor', type=float, default=1.)
    parser.add_argument('--weight-decay', '-wd', type=float, default=0.0, help='Weight decay for AdamW optimizer')
    args = parser.parse_args()

    print(vars(args))
    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )

    if args.experiment_id is not None:
        assert 520_000 <= args.experiment_id < 530_000, args.experiment_id
        results = ResultsJSON(eid=args.experiment_id, path='./results/')
        results.store_args(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)
    model, loss_fn, optim = get_model(args)

    best_acc = 0

    try:
        for i, (x, y) in tqdm(enumerate(load_n(train_loader, args.num_iterations)),
                              desc='iteration', total=args.num_iterations):
            log_memory()
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
            x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
            y = y.to('cuda')
            loss_val = train(model, x, y, loss_fn, optim)
            
            if (i+1) % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                log_memory()
            
            if (i+1) % args.eval_freq == 0 and i >= 50:
                torch.cuda.empty_cache()
                gc.collect()
                log_memory()
                if args.extensive_eval and validation_loader is not None:
                    train_acc_train_mode = eval_model(model, train_loader, mode=True, max_batches=10)
                    valid_acc_eval_mode = eval_model(model, validation_loader, mode=False, max_batches=10)
                    valid_acc_train_mode = eval_model(model, validation_loader, mode=True, max_batches=10)
                else:
                    train_acc_train_mode = valid_acc_eval_mode = valid_acc_train_mode = -1
                train_acc_eval_mode = eval_model(model, train_loader, mode=False, max_batches=10)
                test_acc_eval_mode = eval_model(model, test_loader, mode=False, max_batches=10)
                test_acc_train_mode = eval_model(model, test_loader, mode=True, max_batches=10)
                r = {
                    'train_acc_eval_mode': train_acc_eval_mode,
                    'train_acc_train_mode': train_acc_train_mode,
                    'valid_acc_eval_mode': valid_acc_eval_mode,
                    'valid_acc_train_mode': valid_acc_train_mode,
                    'test_acc_eval_mode': test_acc_eval_mode,
                    'test_acc_train_mode': test_acc_train_mode,
                    'loss': loss_val,
                }
                print(r)
                if valid_acc_eval_mode > best_acc:
                    best_acc = valid_acc_eval_mode
                    print("New best validation accuracy!")
                    if args.experiment_id is not None:
                        results.store_final_results(r)
                if args.experiment_id is not None:
                    results.save()

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        final_model_path = f"model_interrupted_{args.dataset}_{args.num_layers}layers_{args.num_neurons}neurons.pt"
        torch.save(model.state_dict(), final_model_path)
        print(f"Model saved to {final_model_path}")
        exit(0)

    final_model_path = f"model_final_{args.dataset}_{args.num_layers}layers_{args.num_neurons}neurons.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to {final_model_path}")

    if args.packbits_eval:
        print("\nMeasuring inference time with PackBitsTensor (averaged over 10 runs) ...")
        model.eval()
        inference_times = []
        sample_batch, _ = next(iter(test_loader))
        sample_batch = sample_batch.to('cuda').reshape(sample_batch.shape[0], -1).round().bool()
        with torch.no_grad():
            for _ in range(10):
                sample_batch_packed = PackBitsTensor(sample_batch)
                _ = model(sample_batch_packed)
        for _ in range(10):
            sample_batch_packed = PackBitsTensor(sample_batch)
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                _ = model(sample_batch_packed)
            torch.cuda.synchronize()
            inference_times.append(time.time() - start_time)
        avg_time = sum(inference_times) / len(inference_times)
        print("Average inference time (PackBitsTensor) over 10 runs for one batch: {:.6f} seconds".format(avg_time))
        print(f"Average inference time per sample: {avg_time / sample_batch.shape[0]:.6f} seconds")

    if args.compile_model:
        print('\n' + '='*80)
        print(' Converting the model to C code and compiling it...')
        print('='*80)
        for opt_level in range(4):
            for num_bits in [64]:
                os.makedirs('lib', exist_ok=True)
                save_lib_path = 'lib/{:08d}_{}.so'.format(
                    args.experiment_id if args.experiment_id is not None else 0, num_bits
                )
                compiled_model = CompiledLogicNet(
                    model=model,
                    num_bits=num_bits,
                    cpu_compiler='gcc',
                    verbose=True,
                )
                compiled_model.compile(
                    opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
                    save_lib_path=save_lib_path,
                    verbose=True
                )
                correct, total = 0, 0
                with torch.no_grad():
                    for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
                        data = torch.nn.Flatten()(data).bool().numpy()
                        output = compiled_model(data, verbose=True)
                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]
                acc3 = correct / total
                print('COMPILED MODEL', num_bits, acc3)