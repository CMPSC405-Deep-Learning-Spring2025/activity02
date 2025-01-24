import numpy as np
import torch

# Task 1: Dataset Creation
np.random.seed(42)  # For reproducibility
dataset = np.random.rand(5, 3)  # Creating a 5x3 dataset
print("Original Dataset:")
print(dataset)

# Task 2: Normalization
def normalize(data):
    mean = data.mean(axis=0)  # Compute mean along columns
    std = data.std(axis=0)    # Compute standard deviation along columns
    return (data - mean) / std

normalized_dataset = normalize(dataset)
print("\nNormalized Dataset:")
print(normalized_dataset)

# Task 3: PyTorch Integration
tensor_dataset = torch.tensor(normalized_dataset, dtype=torch.float32)
print("\nPyTorch Tensor:")
print(tensor_dataset)

# Task 4: Additional Practice: Data Manipulation
# Create random arrays and tensors
np_random_array = np.random.rand(3, 3)
torch_random_tensor = torch.rand(3, 3)

# Perform element-wise multiplication
np_elementwise = normalized_dataset[:, :3] * np_random_array
torch_elementwise = tensor_dataset[:, :3] * torch_random_tensor

# Perform matrix multiplication
np_matrix_mul = normalized_dataset[:, :3] @ np_random_array
torch_matrix_mul = tensor_dataset[:, :3] @ torch_random_tensor

print("\nElement-wise Multiplication (NumPy):")
print(np_elementwise)
print("\nElement-wise Multiplication (PyTorch):")
print(torch_elementwise)

print("\nMatrix Multiplication (NumPy):")
print(np_matrix_mul)
print("\nMatrix Multiplication (PyTorch):")
print(torch_matrix_mul)
n
