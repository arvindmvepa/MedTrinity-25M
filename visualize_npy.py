import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

def visualize_npy_volume(npy_file, output_dir=None):
    """
    Load an .npy file and save each slice as a JPEG image.
    
    Args:
        npy_file: Path to the .npy file
        output_dir: Directory to save JPEG files (default: same as input file)
    """
    # Load the .npy file
    try:
        volume = np.load(npy_file)
        print(f"Loaded volume with shape: {volume.shape}")
    except Exception as e:
        print(f"Error loading {npy_file}: {e}")
        return
    
    # Handle different dimensionalities
    if volume.ndim == 4 and volume.shape[0] == 1:
        # Remove channel dimension: (1, D, H, W) -> (D, H, W)
        volume = volume.squeeze(0)
    elif volume.ndim == 3:
        # Already in (D, H, W) format
        pass
    else:
        print(f"Unexpected volume shape: {volume.shape}")
        return
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(npy_file).parent / f"{Path(npy_file).stem}_slices"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    print(f"Saving slices to: {output_dir}")
    
    # Save each slice as JPEG
    num_slices = volume.shape[0]
    for i in range(num_slices):
        slice_data = volume[i]  # Shape: (H, W)
        
        # Normalize to 0-255 for visualization
        slice_normalized = ((slice_data - slice_data.min()) / 
                           (slice_data.max() - slice_data.min() + 1e-8) * 255).astype(np.uint8)
        
        # Create figure and save
        plt.figure(figsize=(8, 8))
        plt.imshow(slice_normalized, cmap='gray')
        plt.title(f"Slice {i:03d}/{num_slices-1:03d}")
        plt.axis('off')
        
        output_file = output_dir / f"slice_{i:03d}.jpg"
        plt.savefig(output_file, bbox_inches='tight', dpi=100, format='jpg')
        plt.close()
    
    print(f"Saved {num_slices} slices to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize .npy volume files by saving slices as JPEGs")
    parser.add_argument("npy_file", help="Path to the .npy file to visualize")
    parser.add_argument("--output_dir", help="Output directory for JPEG files (default: next to input file)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npy_file):
        print(f"File not found: {args.npy_file}")
        return
    
    visualize_npy_volume(args.npy_file, args.output_dir)

if __name__ == "__main__":
    main()