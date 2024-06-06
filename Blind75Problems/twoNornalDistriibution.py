import numpy as np
import matplotlib.pyplot as plt

# Parameters
mean = 10  # Mean for both distributions
variance =.05   # Variance for both distributions
std_dev = np.sqrt(variance)  # Standard deviation

# Generate random samples for two normal distributions
np.random.seed(42)  # Setting seed for reproducibility
distribution_1 = np.random.normal(mean, std_dev, 10000)
distribution_2 = np.random.normal(mean, std_dev, 10000)

# Subtract the distributions
subtracted_distribution = distribution_1 - distribution_2

# Plotting
plt.figure(figsize=(8, 6))

# Histogram of subtracted distribution
plt.hist(subtracted_distribution, bins=300, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Subtraction of Two Normal Distributions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
