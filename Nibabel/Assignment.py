import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def normalize_volume(volume):
    """
    Normalizes the 3D volume using min-max normalization.
    Returns the new array with values between 0 and 1.
    """
    v_min = np.min(volume)
    v_max = np.max(volume)
    
    # Avoid division by zero
    if v_max - v_min == 0:
        return volume
        
    normalized_volume = (volume - v_min) / (v_max - v_min)
    return normalized_volume

def plot_slices(volume, indices):
    """
    Plots specific slice numbers from a 3D volume.
    """
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(20, 5))
    
    for i, idx in enumerate(indices):
        # Taking a slice along the axial plane (z-axis)
        axes[i].imshow(volume[:, :, idx], cmap='gray')
        axes[i].set_title(f'Slice: {idx}')
        axes[i].axis('off')
    
    plt.show()

# Load another MRI file (ensure you have a .nii or .nii.gz file)
file_path = 'CT_AVM.nii.gz'
img = nib.load(file_path)

# Convert to a numpy array (Volume)
volume_data = img.get_fdata()

# Step 1: Normalize the entire volume
normalized_data = normalize_volume(volume_data)

# Step 2: Visualize 5 slices
# We choose 5 equally spaced indices across the volume
total_slices = normalized_data.shape[2]
indices_to_plot = np.linspace(0, total_slices - 1, 5, dtype=int)

plot_slices(normalized_data, indices_to_plot)





