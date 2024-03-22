import numpy as np

# Generate random heights and weights for 50 individuals
np.random.seed(0)  # for reproducibility
heights = np.random.normal(170, 10, 50)  # mean=170, std=10
weights = np.random.normal(70, 15, 50)   # mean=70, std=15

# Calculating descriptive statistics
height_mean = np.mean(heights)
height_median = np.median(heights)
height_std = np.std(heights)
height_range = np.max(heights) - np.min(heights)

weight_mean = np.mean(weights)
weight_median = np.median(weights)
weight_std = np.std(weights)
weight_range = np.max(weights) - np.min(weights)

print("Height Mean:", height_mean)
print("Height Median:", height_median)
print("Height Standard Deviation:", height_std)
print("Height Range:", height_range)

print("\nWeight Mean:", weight_mean)
print("Weight Median:", weight_median)
print("Weight Standard Deviation:", weight_std)
print("Weight Range:", weight_range)


# Define the event (e.g., probability of being taller than 175 cm)
threshold_height = 175
probability_taller = np.sum(heights > threshold_height) / len(heights)
print("Probability of being taller than", threshold_height, "cm:", probability_taller)


import matplotlib.pyplot as plt

# Plot histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(heights, bins=10, color='skyblue', edgecolor='black')
plt.title('Height Distribution')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(weights, bins=10, color='salmon', edgecolor='black')
plt.title('Weight Distribution')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Sampling and plotting sample means distribution
sample_means = []
num_samples = 1000
sample_size = 30

for _ in range(num_samples):
    sample = np.random.choice(heights, size=sample_size, replace=False)
    sample_means.append(np.mean(sample))

plt.hist(sample_means, bins=30, color='lightgreen', edgecolor='black')
plt.title('Distribution of Sample Means (CLT)')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.show()

from scipy.stats import t

confidence_level = 0.95
degrees_of_freedom = len(heights) - 1
height_std_error = height_std / np.sqrt(len(heights))
t_critical = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
margin_of_error = t_critical * height_std_error

height_ci = (height_mean - margin_of_error, height_mean + margin_of_error)
print("95% Confidence Interval for Height Mean:", height_ci)

from scipy.stats import ttest_1samp

# Hypothesis: Testing if the mean height is different from 170 cm
null_hypothesis = 170
t_stat, p_value = ttest_1samp(heights, null_hypothesis)
print("t-statistic:", t_stat)
print("p-value:", p_value)

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Mean height is different from", null_hypothesis, "cm.")
else:
    print("Fail to reject the null hypothesis: Mean height is not significantly different from", null_hypothesis, "cm.")
