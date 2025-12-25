import numpy as np

# 1. Create an array of length 20 modeling a Poisson distribution
# We'll use lambda (lam) = 10 as the expected number of occurrences
poisson_array = np.random.poisson(lam=10, size=20)
print("Original Poisson Array:\n", poisson_array)

# 2. Calculate the mean and standard deviation for the transformation
mean_val = np.mean(poisson_array)
std_val = np.std(poisson_array)

print(f"\nMean: {mean_val}")
print(f"Standard Deviation: {std_val}")

# 3. Center the array around the mean
# This is done by subtracting the mean from each element
centered_array = poisson_array - mean_val
print("\nCentered Array (Element - Mean):\n", centered_array)

# 4. Normalise the array by the standard deviation
# This is done by dividing the centered elements by the standard deviation
normalized_array = centered_array / std_val
print("\nNormalized Array (Centered / Std Dev):\n", normalized_array)
