import numpy as np
import matplotlib.pyplot as plt

# 1. Create a numpy array of length 1000 resembling a normal distribution
# loc=0 (mean), scale=1 (standard deviation)
normal_data = np.random.normal(loc=0, scale=1, size=1000)

# 2. Create a numpy array for the Beta distribution (a=2, b=5)
beta_data = np.random.beta(a=2, b=5, size=1000)

# 3. Create 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Beta Distribution Histogram
ax1.hist(beta_data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_title('Beta Distribution (a=2, b=5)')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

# Plot 2: Normal Distribution Histogram
ax2.hist(normal_data, bins=20, color='salmon', edgecolor='black', alpha=0.7)
ax2.set_title('Normal Distribution (mean=0, std=1)')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')

# Adjust layout and save/show the plot
plt.tight_layout()
plt.savefig('distribution_comparison.png')
plt.show()
