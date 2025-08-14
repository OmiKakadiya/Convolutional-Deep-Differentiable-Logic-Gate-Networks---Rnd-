import re

def parse_eval_accuracies(filename):
    with open(filename, 'r', encoding='utf-8') as f:  # try utf-8-sig or ISO-8859-1 if this fails
        content = f.read()

    # Extract train and test eval mode accuracies
    train_pattern = re.compile(r"'train_acc_eval_mode':\s*([0-9.]+)")
    test_pattern = re.compile(r"'test_acc_eval_mode':\s*([0-9.]+)")

    train_matches = [float(match) for match in train_pattern.findall(content)]
    test_matches = [float(match) for match in test_pattern.findall(content)]

    def summarize(accuracies):
        if not accuracies:
            return {'last': None, 'max': None, 'average': None}
        return {
            'last': accuracies[-1],
            'max': max(accuracies),
            'average': sum(accuracies) / len(accuracies)
        }

    return {
        'train_acc_eval_mode': summarize(train_matches),
        'test_acc_eval_mode': summarize(test_matches),
    }

# Example usage
file_path = 'aa1.txt'
results = parse_eval_accuracies(file_path)
if results:
    print("Train Accuracy (Eval Mode):")
    print("  Last:", results['train_acc_eval_mode']['last'])
    print("  Max:", results['train_acc_eval_mode']['max'])
    print("  Average:", results['train_acc_eval_mode']['average'])

    print("\nTest Accuracy (Eval Mode):")
    print("  Last:", results['test_acc_eval_mode']['last'])
    print("  Max:", results['test_acc_eval_mode']['max'])
    print("  Average:", results['test_acc_eval_mode']['average'])
