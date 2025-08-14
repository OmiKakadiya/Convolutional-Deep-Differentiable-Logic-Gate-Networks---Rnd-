import argparse
import math
import random
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import itertools
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}

class LogicLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grad_factor=1.0, connections='unique'):
        super(LogicLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_neurons = out_dim
        self.num_weights = out_dim * 16
        self.grad_factor = grad_factor
        # Weights for operator probabilities
        self.weights = nn.Parameter(torch.randn(out_dim, 16) * 0.1)
        # Connections: each neuron takes 2 inputs
        self.connections = (nn.Parameter(torch.randint(0, in_dim, (out_dim, 2)).long())
                           if connections == 'random' else
                           torch.randint(0, in_dim, (out_dim, 2)).long())
        # 16 logic operators as per Table 1 (probabilistic T-norms/T-conorms)
        self.operators = [
            lambda a, b: torch.zeros_like(a),  # False
            lambda a, b: a * b,  # AND
            lambda a, b: a * (1 - b),  # A and not B
            lambda a, b: a,  # A
            lambda a, b: (1 - a) * b,  # not A and B
            lambda a, b: b,  # B
            lambda a, b: a + b - 2 * a * b,  # XOR
            lambda a, b: a + b - a * b,  # OR
            lambda a, b: 1 - (a + b - a * b),  # NOR
            lambda a, b: 1 - (a + b - 2 * a * b),  # XNOR
            lambda a, b: 1 - b,  # not B
            lambda a, b: a + (1 - b) - a * (1 - b),  # A or not B
            lambda a, b: 1 - a,  # not A
            lambda a, b: (1 - a) + b - (1 - a) * b,  # not A or B
            lambda a, b: 1 - a * b,  # NAND
            lambda a, b: torch.ones_like(a),  # True
        ]

    def forward(self, x, return_gate_indices=False):
        batch_size = x.shape[0]
        # Softmax for operator probabilities
        logits = self.weights / self.grad_factor
        probs = torch.softmax(logits, dim=-1)  # [out_dim, 16]
        # Gather inputs
        inp1 = x[:, self.connections[:, 0]]  # [batch_size, out_dim]
        inp2 = x[:, self.connections[:, 1]]  # [batch_size, out_dim]
        # Compute operator outputs
        out = torch.zeros(batch_size, self.out_dim, device=x.device, dtype=torch.float32)
        for op_idx, op in enumerate(self.operators):
            op_out = op(inp1, inp2).float()  # Ensure float for AMP
            out += probs[:, op_idx].view(1, -1) * op_out
        if return_gate_indices:
            gate_indices = torch.argmax(probs, dim=-1)  # [out_dim]
            return out, gate_indices
        return out

class GroupSum(nn.Module):
    def __init__(self, num_classes, tau):
        super(GroupSum, self).__init__()
        self.num_classes = num_classes
        self.tau = tau

    def forward(self, x):
        batch_size = x.shape[0]
        neurons_per_class = x.shape[1] // self.num_classes
        x = x.view(batch_size, self.num_classes, neurons_per_class)
        out = x.sum(dim=-1) / self.tau
        return out

def load_dataset(args):
    validation_loader = None
    if args.dataset in ['mnist', 'mnist20x20']:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: (x > 0.5).float()),
        ])
        train_set = torchvision.datasets.MNIST('./data-mnist', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST('./data-mnist', train=False, transform=transform)
        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    elif 'cifar-10' in args.dataset:
        transform = {
            'cifar-10-3-thresholds': lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0),
            'cifar-10-31-thresholds': lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0),
        }[args.dataset]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(transform),
        ])
        train_set = torchvision.datasets.CIFAR10('./data-cifar', train=True, download=True, transform=transforms)
        test_set = torchvision.datasets.CIFAR10('./data-cifar', train=False, transform=transforms)
        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    else:
        raise NotImplementedError(f'The dataset {args.dataset} is not supported!')
    return train_loader, validation_loader, test_loader

def input_dim_of_dataset(dataset):
    return {
        'mnist': 784,
        'mnist20x20': 400,
        'cifar-10-3-thresholds': 3 * 32 * 32 * 3,
        'cifar-10-31-thresholds': 3 * 32 * 32 * 31,
    }[dataset]

def num_classes_of_dataset(dataset):
    return {
        'mnist': 10,
        'mnist20x20': 10,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }[dataset]

def get_model(args, device='cuda'):
    llkw = dict(grad_factor=args.grad_factor, connections=args.connections)
    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)
    logic_layers = []
    arch = args.architecture
    k = args.num_neurons
    l = args.num_layers
    if arch == 'randomly_connected':
        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
        for _ in range(l - 1):
            logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))
        model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, args.tau)
        )
    else:
        raise NotImplementedError(arch)
    total_num_neurons = sum(layer.num_neurons for layer in logic_layers if isinstance(layer, LogicLayer))
    total_num_weights = sum(layer.num_weights for layer in logic_layers if isinstance(layer, LogicLayer))
    print(f'total_num_neurons={total_num_neurons}')
    print(f'total_num_weights={total_num_weights}')
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Total trainable parameters: {param_count}')
    if param_count == 0:
        raise ValueError("Model has no trainable parameters!")
    print(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return model, loss_fn, optimizer

def train(model, x, y, loss_fn, optimizer, scaler):
    with autocast():
        x = model(x)
        loss = loss_fn(x, y)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()

def eval(model, loader, mode, device='cuda'):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        res = np.mean(
            [
                (model(x.to(device).round()).argmax(-1) == y.to(device)).float().mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res

def collect_gate_distributions(model, loader, device='cuda'):
    model.eval()
    gate_distributions = {}
    operator_names = [
        'False', 'AND', 'A and not B', 'A', 'not A and B', 'B', 'XOR', 'OR',
        'NOR', 'XNOR', 'not B', 'A or not B', 'not A', 'not A or B', 'NAND', 'True'
    ]
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device).round()
            current_x = x
            layer_idx = 0
            for module in model:
                if isinstance(module, LogicLayer):
                    current_x, gate_indices = module(current_x, return_gate_indices=True)
                    if layer_idx not in gate_distributions:
                        gate_distributions[layer_idx] = np.zeros(16, dtype=np.int64)
                    gate_counts = np.bincount(gate_indices.cpu().numpy().flatten(), minlength=16)
                    gate_distributions[layer_idx] += gate_counts
                    print(f"Layer {layer_idx}, Batch {batch_idx}: Raw counts = {gate_counts.tolist()}")
                    layer_idx += 1
                else:
                    current_x = module(current_x)
    result = {}
    for layer_idx, counts in gate_distributions.items():
        total = counts.sum()
        normalized = counts / total if total > 0 else counts
        print(f"Layer {layer_idx}: Normalized = {normalized.tolist()}")
        result[f'layer_{layer_idx}'] = {
            op_name: float(count) for op_name, count in zip(operator_names, normalized)
        }
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train logic gate network and collect gate distributions.')
    parser.add_argument('--dataset', type=str, default='cifar-10-3-thresholds',
                        choices=['mnist', 'mnist20x20', 'cifar-10-3-thresholds', 'cifar-10-31-thresholds'])
    parser.add_argument('--tau', type=float, default=1/0.03)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--valid_set_size', type=float, default=0.1)
    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
    parser.add_argument('--architecture', type=str, default='randomly_connected')
    parser.add_argument('--num_neurons', type=int, default=12000)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--grad_factor', type=float, default=1.0)
    args = parser.parse_args()

    print(vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    class_count = num_classes_of_dataset(args.dataset)
    if args.num_neurons % class_count != 0:
        raise ValueError(f"num_neurons ({args.num_neurons}) must be divisible by num_classes ({class_count})")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)
    model, loss_fn, optim = get_model(args, device=device)
    scaler = GradScaler()

    # Estimate runtime
    start_time = time.time()
    print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    est_time_per_epoch = len(train_loader) * 0.1 / 3600  # Adjust based on hardware
    est_total_time = args.num_epochs * est_time_per_epoch
    print(f"Estimated runtime: {est_total_time:.2f} hours")

    best_acc = 0
    best_model_path = f"model_best_{args.dataset}_{args.num_layers}layers_{args.num_neurons}neurons.pt"

    for epoch in range(args.num_epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}'):
            x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[32]).to(device)
            y = y.to(device)
            loss = train(model, x, y, loss_fn, optim, scaler)

        if (epoch + 1) % args.eval_freq == 0:
            train_accuracy_eval_mode = eval(model, train_loader, mode=False, device=device)
            valid_accuracy_eval_mode = eval(model, validation_loader, mode=False, device=device)
            test_accuracy_eval_mode = eval(model, test_loader, mode=False, device=device)
            r = {
                'train_acc_eval_mode': train_accuracy_eval_mode,
                'valid_acc_eval_mode': valid_accuracy_eval_mode,
                'test_acc_eval_mode': test_accuracy_eval_mode,
            }
            print(f"Epoch {epoch + 1}: {r}")
            if valid_accuracy_eval_mode > best_acc:
                best_acc = valid_accuracy_eval_mode
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")

    # Save final model
    final_model_path = f"model_final_{args.dataset}_{args.num_layers}layers_{args.num_neurons}neurons.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Collect and save gate distributions
    print("\nCollecting gate distributions...")
    gate_distributions = collect_gate_distributions(model, test_loader, device=device)
    gate_dist_path = f"gate_distributions_{args.dataset}_{args.num_layers}layers_{args.num_neurons}neurons.json"
    with open(gate_dist_path, 'w') as f:
        json.dump(gate_distributions, f, indent=4)
    print(f"Gate distributions saved to {gate_dist_path}")

    # Print runtime
    elapsed_time = (time.time() - start_time) / 3600
    print(f"Actual runtime: {elapsed_time:.2f} hours")