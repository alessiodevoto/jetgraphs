# read back the train_idx_mod
import h5py
import numpy as np
# Plot and save GIs based on Kappa method


import matplotlib.pyplot as plt

# Load previously calculated gradient magnitudes
magnitudes = np.load('gradient_magnitudes_test0corr.npy')


# Load predefined outlier indices
#outlier_indices = remaining_indices#[:100] #np.load('outlier_indices.npy')

# # Filter invalid indices (out of bounds)
# #valid_outlier_indices = [i for i in outlier_indices if i < len(magnitudes)]
# outlier_magnitudes = [magnitudes[i] for i in remaining_indices]
# #gis = [77753, 115616, 31298, 76948, 93288, 145266]
# gi_magnitudes = [magnitudes[i] for i in gis]


# Extract magnitudes of outliers
#outlier_magnitudes = [magnitudes[i] for i in outlier_indices]
Kappa = 3 # 5 or 10
threshold = np.median(magnitudes) + Kappa * (np.percentile(magnitudes, 75) - np.percentile(magnitudes, 25))

# Determine outlier indices based on the predefined threshold
outlier_indices = [i for i, magnitude in enumerate(magnitudes) if magnitude > threshold]

# Save the outlier indices to a file
np.save('outlier_indicesK3corr_test0.npy', outlier_indices)
print('outliers len: ', len(outlier_indices))
#print('outliers indices: ', outlier_indices)

# # Plot histogram of all magnitudes
# plt.hist(magnitudes, bins=1500, edgecolor='black')
# plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1, label=f'Threshold: {threshold:.4f}')
# plt.title('Histogram of Gradient Magnitudes')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency')
# plt.legend()

# # Set x-axis limit to 0.3
# plt.xlim(0, 0.07)

# # # Set x-axis ticks to show 100 divisions in each 0.1 section
# # x_ticks = np.linspace(0, 0.2, 21)
# # plt.xticks(x_ticks)

# plt.savefig('gradient_magnitudes_histogram.pdf')
# plt.show()

# # Plot histogram of outlier magnitudes
# plt.hist(gi_magnitudes, bins=100, edgecolor='black')# gi_magnitudes
# plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1, label=f'Threshold: {threshold:.4f}')
# plt.title('Histogram of tpOppstest1 Gradient thresh k=5 ')
# #plt.title('Histogram of 60 percent FN Props Outliers from Gradient thresh k=550 ')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency')
# plt.legend()
# # Set x-axis limit to 0.3
# #plt.xlim(0, 0.15)

# # Set x-axis ticks to show 100 divisions in each 0.1 section
# #x_ticks = np.linspace(0, 0.2, 21)
# #plt.xticks(x_ticks)
# plt.savefig('outlier_method1tpOppstest1_threshk5_histogram.pdf')
# #plt.savefig('outlier_method1FNProps_gradient_threshk550_histogram.pdf')
# plt.show()