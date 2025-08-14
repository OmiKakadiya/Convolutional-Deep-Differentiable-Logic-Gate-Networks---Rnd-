import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from results_json import ResultsJSON

import mnist_dataset
import uci_datasets

torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}


def load_dataset(args):
    validation_loader = None
    if args.dataset == 'adult':
        train_set = uci_datasets.AdultDataset('./data-uci', split='train', download=True, with_val=False)
        test_set  = uci_datasets.AdultDataset('./data-uci', split='test',  with_val=False)
    elif args.dataset == 'breast_cancer':
        train_set = uci_datasets.BreastCancerDataset('./data-uci', split='train', download=True, with_val=False)
        test_set  = uci_datasets.BreastCancerDataset('./data-uci', split='test',  with_val=False)
    elif args.dataset.startswith('monk'):
        style = int(args.dataset[4])
        train_set = uci_datasets.MONKsDataset('./data-uci', style, split='train', download=True, with_val=False)
        test_set  = uci_datasets.MONKsDataset('./data-uci', style, split='test',  with_val=False)
    elif args.dataset in ['mnist', 'mnist20x20']:
        train_set = mnist_dataset.MNIST('./data-mnist', train=True, download=True, remove_border=args.dataset=='mnist20x20')
        test_set  = mnist_dataset.MNIST('./data-mnist', train=False, remove_border=args.dataset=='mnist20x20')

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=args.batch_size, shuffle=False,
            pin_memory=True, drop_last=True)
    elif 'cifar-10' in args.dataset:
        transform = {
            'cifar-10-real-input': lambda x: x,
            'cifar-10-3-thresholds': lambda x: torch.cat([(x > (i + 1)/4).float() for i in range(3)], dim=0),
            'cifar-10-31-thresholds': lambda x: torch.cat([(x > (i + 1)/32).float() for i in range(31)], dim=0),
        }[args.dataset]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(transform),
        ])
        train_set = torchvision.datasets.CIFAR10('./data-cifar', train=True, download=True, transform=transforms)
        test_set  = torchvision.datasets.CIFAR10('./data-cifar', train=False, transform=transforms)

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

        validation_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=args.batch_size, shuffle=False,
            pin_memory=True, drop_last=True)
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not supported!')

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=(args.dataset in ['mnist','mnist20x20'] or 'cifar-10' in args.dataset),
        drop_last=True, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=(args.dataset in ['mnist','mnist20x20'] or 'cifar-10' in args.dataset),
        drop_last=True, num_workers=4)
    return train_loader, validation_loader, test_loader


def load_n(loader, n):
    i = 0
    while i < n:
        for batch in loader:
            yield batch
            i += 1
            if i == n:
                break


def input_dim_of_dataset(dataset):
    return {
        'adult': 116,
        'breast_cancer': 51,
        'monk1': 17,
        'monk2': 17,
        'monk3': 17,
        'mnist': 784,
        'mnist20x20': 400,
        'cifar-10-real-input': 3 * 32 * 32,
        'cifar-10-3-thresholds': 3 * 32 * 32 * 3,
        'cifar-10-31-thresholds': 3 * 32 * 32 * 31,
    }[dataset]


def num_classes_of_dataset(dataset):
    return {
        'adult': 2,
        'breast_cancer': 2,
        'monk1': 2,
        'monk2': 2,
        'monk3': 2,
        'mnist': 10,
        'mnist20x20': 10,
        'cifar-10-real-input': 10,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }[dataset]


def measure_time(model, device, dummy_input, runs=50):
    model = model.to(device)
    dummy = dummy_input.to(device)
    # warm-up
    for _ in range(10):
        _ = model(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        _ = model(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return (elapsed / runs) * 1000  # ms per image


def compute_and_log_stats(model, args):
    # 1) # parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 2) model size in MB
    param_bytes = num_params * (args.training_bit_count / 8)
    param_MB = param_bytes / (1024**2)
    # 3) FLOPs (2*m*n per Linear layer)
    flops = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            flops += 2 * m.in_features * m.out_features
    # print & store
    print(f"#Params: {num_params:,}")
    print(f"Model Size: {param_MB:.2f} MB")
    print(f"FLOPs: {flops:,}")
    if args.experiment_id is not None:
        results.store_results({
            'num_params': num_params,
            'model_size_MB': param_MB,
            'flops': flops,
        })


def get_model(args):
    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    layers = []
    arch = args.architecture
    k = args.num_neurons
    l = args.num_layers
    total_num_neurons = 0

    if arch == 'fully_connected':
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(in_dim, k,
                        dtype=BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]))
        layers.append(torch.nn.ReLU())
        total_num_neurons += k
        for _ in range(l - 2):
            layers.append(torch.nn.Linear(k, k,
                            dtype=BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]))
            layers.append(torch.nn.ReLU())
            total_num_neurons += k
        layers.append(torch.nn.Linear(k, class_count,
                        dtype=BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]))
        total_num_neurons += class_count
        model = torch.nn.Sequential(*layers)
    else:
        raise NotImplementedError(arch)

    # count neurons (for logic-gate nets, this equals # binary ops)
    print(f"total_num_neurons={total_num_neurons}")
    total_num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_num_weights={total_num_weights}")
    if args.experiment_id is not None:
        results.store_results({
            'total_num_neurons': total_num_neurons,
            'total_num_weights': total_num_weights,
        })

    model = model.to('cuda')
    compute_and_log_stats(model, args)

    print(model)
    if args.experiment_id is not None:
        results.store_results({'model_str': str(model)})

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    return model, loss_fn, optimizer


def train(model, x, y, loss_fn, optimizer):
    out = model(x)
    loss = loss_fn(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def eval(model, loader, mode, timing=False):
    """
    Returns (accuracy, cpu_ms_per_img, gpu_ms_per_img) if timing=True,
    else (accuracy, None, None).
    """
    orig_mode = model.training
    accs = []
    with torch.no_grad():
        model.train(mode=mode)
        for x, y in loader:
            x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
            y = y.to('cuda')
            out = model(x)
            accs.append((out.argmax(-1) == y).float().mean().item())

        cpu_t = gpu_t = None
        if timing:
            dummy = x  # reuse last batch shape
            # CPU timing
            torch.cuda.synchronize()
            start = time.time()
            _ = model(dummy)
            cpu_t = (time.time() - start) * 1000 / dummy.size(0)
            # GPU timing
            torch.cuda.synchronize()
            start = time.time()
            _ = model(dummy)
            torch.cuda.synchronize()
            gpu_t = (time.time() - start) * 1000 / dummy.size(0)

        model.train(mode=orig_mode)

    acc = float(np.mean(accs))
    return (acc, cpu_t, gpu_t) if timing else (acc, None, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train logic gate networks.')
    parser.add_argument('-eid', '--experiment_id', type=int, default=None)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['adult','breast_cancer','monk1','monk2','monk3',
                                 'mnist','mnist20x20',
                                 'cifar-10-real-input','cifar-10-3-thresholds','cifar-10-31-thresholds'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', '-bs', type=int, default=128)
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01)
    parser.add_argument('--training-bit-count', '-c', type=int, default=32)
    parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda','python'])
    parser.add_argument('--num-iterations', '-ni', type=int, default=100000)
    parser.add_argument('--eval-freq', '-ef', type=int, default=2000)
    parser.add_argument('--valid-set-size', '-vss', type=float, default=0.)
    parser.add_argument('--extensive-eval', action='store_true')
    parser.add_argument('--architecture', '-a', type=str, default='fully_connected')
    parser.add_argument('--num_neurons', '-k', type=int, required=True)
    parser.add_argument('--num_layers', '-l', type=int, required=True)
    args = parser.parse_args()

    assert args.num_iterations % args.eval_freq == 0

    if args.experiment_id is not None:
        os.makedirs('./results', exist_ok=True)
        assert 520000 <= args.experiment_id < 530000
        results = ResultsJSON(eid=args.experiment_id, path='./results/')
        results.store_args(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)
    model, loss_fn, optim = get_model(args)

    best_acc = 0.0
    for i, (x, y) in tqdm(enumerate(load_n(train_loader, args.num_iterations)),
                          desc='iteration', total=args.num_iterations):
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
        y = y.to('cuda')
        _ = train(model, x, y, loss_fn, optim)

        if (i + 1) % args.eval_freq == 0:
            # standard eval
            if args.extensive_eval and validation_loader is not None:
                train_acc_t, _, _ = eval(model, train_loader, mode=True, timing=False)
                valid_acc_e, _, _ = eval(model, validation_loader, mode=False, timing=False)
                valid_acc_t, _, _ = eval(model, validation_loader, mode=True, timing=False)
            else:
                train_acc_t = valid_acc_e = valid_acc_t = -1.0

            train_acc_e, _, _ = eval(model, train_loader, mode=False, timing=False)
            test_acc_e, _, _  = eval(model, test_loader, mode=False, timing=False)

            r = {
                'train_acc_eval_mode': train_acc_e,
                'train_acc_train_mode': train_acc_t,
                'valid_acc_eval_mode': valid_acc_e,
                'valid_acc_train_mode': valid_acc_t,
                'test_acc_eval_mode': test_acc_e,
            }
            if args.experiment_id is not None:
                results.store_results(r)
            else:
                print(r)

            if valid_acc_e > best_acc:
                best_acc = valid_acc_e
                if args.experiment_id is not None:
                    results.store_final_results(r)
                else:
                    print('IS THE BEST UNTIL NOW.')

            # now measure real inference time per image on test set
            test_acc, cpu_ms, gpu_ms = eval(model, test_loader, mode=False, timing=True)
            print(f"Test: {test_acc:.2%}, CPU {cpu_ms:.2f} ms/img, GPU {gpu_ms:.2f} ms/img")
            if args.experiment_id is not None:
                results.store_results({
                    'test_time_ms_per_image': cpu_ms,
                    'test_time_gpu_ms_per_image': gpu_ms
                })

            if args.experiment_id is not None:
                results.save()
