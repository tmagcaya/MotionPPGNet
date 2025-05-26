import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset_sample(dataset, sample_idx):
    # Retrieve a sample from the dataset
    ppg, acc, ground_truth = dataset[sample_idx]
    ppg = np.array(ppg.squeeze())
    acc = np.array(acc.squeeze())
    ground_truth = np.array(ground_truth.squeeze())
    label = f"Sample idx = {sample_idx}, BPM = {ground_truth}"
    # Plotting  
    plt.figure(figsize=(8, 4))

    # Plot PPG signal
    plt.subplot(2, 1, 1)
    plt.plot(ppg, label=label)
    plt.title('PPG Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot Accelerometer Magnitude
    plt.subplot(2, 1, 2)
    plt.plot(acc, label=label)
    plt.title('Accelerometer Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.tight_layout()
    plt.show()
