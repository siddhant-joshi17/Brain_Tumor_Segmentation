import numpy as np

# 1. Create an array of 18 random elements of shape (3, 6)
arr = np.random.rand(3, 6)
print("Original random array (3, 6):\n", arr)

# 2. Add the array [0, 0, 2, 4, 5, 3] to each row (Broadcasting)
add_arr = np.array([0, 0, 2, 4, 5, 3])
arr = arr + add_arr

# 3. Reshape it to a (9, 2) array
reshaped_arr = arr.reshape(9, 2)

# 4. Take its transpose (Resulting shape: 2, 9)
final_arr = reshaped_arr.T
print("\nFinal Transposed Array (2, 9):\n", final_arr)

# 5. Find the locations where elements are greater than the mean
mean_val = np.mean(final_arr)
locations = np.argwhere(final_arr > mean_val)

print(f"\nMean of the array: {mean_val:.4f}")
print("\nLocations (indices) where elements are greater than the mean:")
print(locations)