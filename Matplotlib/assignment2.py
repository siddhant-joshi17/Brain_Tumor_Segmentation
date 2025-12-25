import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a NumPy array of 1000 random values from a normal distribution (mean=0, std=1)
np.random.seed(42)  # Seed for reproducibility
data = np.random.normal(loc=0, scale=1, size=1000)

# 2. Calculate the mean, 25th percentile, and 75th percentile
mean_val = np.mean(data)
p25 = np.percentile(data, 25)
p75 = np.percentile(data, 75)

# 3. Create the visualization
plt.figure(figsize=(10, 6))

# Plot the histogram with 20 bins and a light pink color scheme
plt.hist(data, bins=20, color='lightpink', edgecolor='white', label='Normal Distribution Data')

# Add vertical lines to indicate key statistics
plt.axvline(mean_val, color='hotpink', linestyle='--', linewidth=2, label=f'Mean ({mean_val:.2f})')
plt.axvline(p25, color='palevioletred', linestyle=':', linewidth=2, label=f'25th Percentile ({p25:.2f})')
plt.axvline(p75, color='deeppink', linestyle=':', linewidth=2, label=f'75th Percentile ({p75:.2f})')

# Annotate the plot
y_limit = plt.ylim()[1]  # Get the max y-value for positioning text
plt.text(mean_val, y_limit * 0.9, ' Mean', color='hotpink', fontweight='bold')
plt.text(p25, y_limit * 0.8, ' 25th Perc', color='palevioletred', fontweight='bold', ha='right')
plt.text(p75, y_limit * 0.8, ' 75th Perc', color='deeppink', fontweight='bold')

# Customize titles, labels, and legend
plt.title('Customized Histogram of Normal Distribution', fontsize=16, color='mediumvioletred')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Save the visualization
plt.savefig('normal_dist_histogram.png')