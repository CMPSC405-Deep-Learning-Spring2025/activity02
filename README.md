# Activity 2

## Due: 9am on January 24, 2025

## Objectives

- Apply transformations to a dataset using PyTorch and NumPy.
- Learn to normalize and manipulate features programmatically.
- Implement a reusable transformation pipeline for datasets.

## Tasks

1. **Dataset Creation**:
   - Create a synthetic dataset with random values using NumPy.
   - Ensure the dataset has at least 5 samples and 3 features.
   - Print the first 5 samples of the dataset.

2. **Normalization**:
   - Implement a function to normalize the features using NumPy.
   - Normalization should be performed using the formula:

     ```
     Normalized Value = (Value - Mean) / Standard Deviation
     ```

   - Apply the normalization function to the dataset.
   - Print the normalized dataset.

3. **PyTorch Integration**:
   - Convert the normalized NumPy dataset to a PyTorch tensor as `torch.tensor(normalized_dataset, dtype=torch.float32)`.
   - Print the PyTorch tensor.

4. **Additional Practice: Data Manipulation**:
   - To further understand the manipulation of arrays and tensors, create a 3x3 NumPy array and a 3x3 PyTorch tensor with random values (i.e., `torch.rand`).
   - Perform element-wise multiplication between the normalized dataset and the random arrays/tensors.
   - Perform matrix multiplication between the normalized dataset and the random arrays/tensors.
   - Print the results.
