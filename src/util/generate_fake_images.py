import os
import argparse
from PIL import Image

def invert_colors(image_path, output_path):
    try:
        with Image.open(image_path) as img:
            inverted_img = Image.eval(img, lambda x: 255 - x)
            inverted_img.save(output_path)
            print(f"Inverted and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def generate_fake_images(real_images_dir, fake_images_dir):
    os.makedirs(fake_images_dir, exist_ok=True)
    for filename in os.listdir(real_images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            real_image_path = os.path.join(real_images_dir, filename)
            fake_image_path = os.path.join(fake_images_dir, filename)
            invert_colors(real_image_path, fake_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 'fake' images by inverting colors from 'real' images.")
    parser.add_argument("--real_images_dir", type=str, required=True,
                        help="Path to the directory containing real images.")
    parser.add_argument("--fake_images_dir", type=str, required=True,
                        help="Path to the directory where fake images will be saved.")
    args = parser.parse_args()

    generate_fake_images(args.real_images_dir, args.fake_images_dir)
