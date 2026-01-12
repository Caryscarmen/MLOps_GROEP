import matplotlib
# Force "Agg" backend so it works on the cluster without a monitor
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from src.ml_core.data.pcam import PCAMDataset

# --- CONFIGURATION ---
# The path you confirmed earlier
BASE_PATH = "/scratch-shared/scur2395/surfdrive"
X_PATH = f"{BASE_PATH}/camelyonpatch_level_2_split_train_x.h5"
Y_PATH = f"{BASE_PATH}/camelyonpatch_level_2_split_train_y.h5"

def run_eda():
    print(f"1. Loading dataset from {BASE_PATH}...")
    print("   (This involves filtering black/white slides, so it might take 30-60 seconds...)")
    
    # We use the dataset class you fixed to ensure we only look at valid data
    ds = PCAMDataset(X_PATH, Y_PATH, filter_data=True)
    print(f"   Dataset loaded. Valid samples: {len(ds)}")

    # -------------------------------------------------------
    # PLOT 1: Class Balance
    # -------------------------------------------------------
    print("2. Generating Class Balance Plot...")
    
    # We need to peek at the labels. Accessing H5 randomly is slow, 
    # but we only need one byte per sample.
    labels = []
    # Use a small subset if it's too slow, but for the report, full stats are better.
    # We iterate over ds.indices to ensure we skip the filtered outliers.
    for i in ds.indices:
        labels.append(ds.y_data[i][0,0,0])
        
    counts = np.bincount(labels)
    # counts[0] is Normal, counts[1] is Tumor
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Normal (0)', 'Tumor (1)'], counts, color=['#4e79a7', '#e15759'])
    plt.title(f'Class Distribution (Total: {len(labels)})')
    plt.ylabel('Count')
    plt.bar_label(bars)
    
    plt.savefig('eda_class_balance.png')
    plt.close()
    print("   Saved -> eda_class_balance.png")

    # -------------------------------------------------------
    # PLOT 2: Pixel Intensity Distribution
    # -------------------------------------------------------
    print("3. Generating Pixel Intensity Plot...")
    
    # Check average intensity of first 1000 VALID images to save time
    intensities = []
    num_samples = min(1000, len(ds))
    
    for i in ds.indices[:num_samples]:
        img = ds.x_data[i]
        intensities.append(img.mean())

    plt.figure(figsize=(6, 4))
    plt.hist(intensities, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.title(f'Pixel Intensity Distribution (Sample of {num_samples})')
    plt.xlabel('Mean Pixel Value (0-255)')
    plt.ylabel('Frequency')
    
    plt.savefig('eda_intensities.png')
    plt.close()
    print("   Saved -> eda_intensities.png")

    # -------------------------------------------------------
    # PLOT 3: Sample Images (5 Normal, 5 Tumor)
    # -------------------------------------------------------
    print("4. Generating Sample Grid...")
    
    # Find indices for first 5 normals and first 5 tumors
    neg_idxs = [i for i, x in enumerate(labels) if x == 0][:5]
    pos_idxs = [i for i, x in enumerate(labels) if x == 1][:5]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Plot Normals (Top Row)
    for i, idx_in_list in enumerate(neg_idxs):
        # map list index back to file index
        real_idx = ds.indices[idx_in_list]
        raw_img = ds.x_data[real_idx] # Get raw uint8 image
        
        axes[0, i].imshow(raw_img.astype('uint8'))
        axes[0, i].set_title("Normal (0)")
        axes[0, i].axis('off')

    # Plot Tumors (Bottom Row)
    for i, idx_in_list in enumerate(pos_idxs):
        real_idx = ds.indices[idx_in_list]
        raw_img = ds.x_data[real_idx]
        
        axes[1, i].imshow(raw_img.astype('uint8'))
        axes[1, i].set_title("Tumor (1)")
        axes[1, i].axis('off')

    plt.suptitle("PCAM Dataset Samples")
    plt.tight_layout()
    plt.savefig('eda_samples.png')
    plt.close()
    print("   Saved -> eda_samples.png")
    
    print("\nDone! Use 'ls' to see your png files.")

if __name__ == "__main__":
    run_eda()