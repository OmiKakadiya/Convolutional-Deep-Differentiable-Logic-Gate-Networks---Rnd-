import ast

# Replace with your actual filename
filename = "mnist_baseline2.txt"

train_acc_vals = []
test_acc_vals = []

with open(filename, "r", encoding="utf-8", errors="ignore") as f:
    # Process lines
    for line in f:
        # Find the start of the dictionary in the line
        start = line.find("{")
        if start != -1:
            data_str = line[start:].strip()
            try:
                # Safely evaluate the dictionary string
                data = ast.literal_eval(data_str)
                # Append values if they are valid (not -1)
                if data.get('train_acc_eval_mode', -1) != -1:
                    train_acc_vals.append(data['train_acc_eval_mode'])
                if data.get('test_acc_eval_mode', -1) != -1:
                    test_acc_vals.append(data['test_acc_eval_mode'])
            except Exception as e:
                print(f"Error parsing line: {line}\nException: {e}")

# Compute averages if lists are not empty
if train_acc_vals:
    avg_train = sum(train_acc_vals) / len(train_acc_vals)
else:
    avg_train = None

if test_acc_vals:
    avg_test = sum(test_acc_vals) / len(test_acc_vals)
else:
    avg_test = None

print("Average train_acc_eval_mode:", avg_train)
print("Average test_acc_eval_mode:", avg_test)
print(len(train_acc_vals))
print(len(test_acc_vals))
