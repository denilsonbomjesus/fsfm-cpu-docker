
import os
import shutil
import argparse
from torchvision.datasets import CelebA

def prepare_celeba_sample(root_dir, dest_dir, num_images):
    """
    Downloads the CelebA dataset and creates a smaller sample from it.
    """
    # Step 1: Download the dataset using torchvision
    print("Downloading CelebA dataset. This may take a while...")
    try:
        celeba_dataset = CelebA(root=root_dir, split='all', download=True)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download CelebA. Error: {e}")
        print("Please check your internet connection or try again later.")
        return

    # The actual images are in a subfolder, typically 'celeba'
    source_img_folder = os.path.join(celeba_dataset.root, celeba_dataset.base_folder, "img_align_celeba")

    if not os.path.isdir(source_img_folder):
        print(f"Error: Could not find the image folder at {source_img_folder}")
        return

    # Step 2: Create destination directories
    train_dir = os.path.join(dest_dir, 'train', 'images')
    val_dir = os.path.join(dest_dir, 'val', 'images')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Step 3: Get a list of image files and sample them
    image_files = sorted(os.listdir(source_img_folder))
    if len(image_files) < num_images:
        print(f"Warning: The dataset contains only {len(image_files)} images, which is less than the requested {num_images}.")
        num_images = len(image_files)

    sample_files = image_files[:num_images]

    # Step 4: Copy sampled images to train and val directories
    print(f"Copying {num_images} images to the sample directory...")
    for i, filename in enumerate(sample_files):
        source_path = os.path.join(source_img_folder, filename)
        
        # Simple split: first 80% to train, last 20% to val
        if i < int(num_images * 0.8):
            dest_path = os.path.join(train_dir, filename)
        else:
            dest_path = os.path.join(val_dir, filename)
        
        shutil.copy(source_path, dest_path)

    print(f"Successfully created a sample of {num_images} images at {dest_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare a sample of the CelebA dataset.")
    parser.add_argument("--root_dir", type=str, default="/app/temp_celeba", help="Temporary directory to download the full CelebA dataset.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory for the sampled dataset.")
    parser.add_argument("--num_images", type=int, default=500, help="Number of images to include in the sample.")
    
    args = parser.parse_args()

    prepare_celeba_sample(args.root_dir, args.dest_dir, args.num_images)

if __name__ == '__main__':
    main()
