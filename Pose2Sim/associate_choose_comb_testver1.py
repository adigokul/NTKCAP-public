import cupy as cp


# """
# Find rows where at least one element in a column is not unique or is not the first occurrence.

# Parameters:
#     array (cupy.ndarray): Input 2D array.

# Returns:
#     cupy.ndarray: Unique row indices where the condition is satisfied.
# """
array = cp.array([
    [2, 0, 1, 1, 4],
    [0, 4, 1, 1, 2],
    [0, 1, 3, 0, 0],
    [3, 1, 2, 3, 3],
    [3, 1, 4, 3, 4]])
rows, cols = array.shape

# Create an array of row indices and column indices
row_indices = cp.arange(rows)
col_indices = cp.arange(cols).repeat(rows).reshape(cols, rows).T

# Combine column indices and values to ensure uniqueness is evaluated per column
col_value_pairs = cp.stack([col_indices.ravel(), array.ravel()], axis=1)

unique_pairs, first_indices, counts = cp.unique(col_value_pairs, axis=0, return_index=True, return_counts=True)
col_value_pairs[first_indices] = cp.array([-1,-1])

print(unique_pairs)
import ipdb;ipdb.set_trace()


# Identify (column, value) pairs that are duplicates (counts > 1)
duplicate_mask = counts > 1

# Find the indices of non-first occurrences for duplicate pairs
duplicate_pairs = unique_pairs[duplicate_mask]
duplicate_indices = cp.where(cp.isin(col_value_pairs, duplicate_pairs).all(axis=1))[0]

# Map duplicate indices back to rows
duplicate_rows = cp.unique(duplicate_indices % rows)



# # Example 5x5 array


# # Display the results
# print("Original Array (on GPU):")
# print(array.get())  # Convert CuPy array to NumPy for display

# print("\nRows with at least one non-unique or non-first element:")
# print(result_rows.get().tolist())  # Convert to NumPy for display






