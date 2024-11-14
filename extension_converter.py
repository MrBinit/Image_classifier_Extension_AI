import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import pyvips

# Set the root directory to process
root_directory = "/home/binit/classifier/data/raw"

# Helper function to convert pyvips image to Pillow-compatible format
def vips_to_pillow(img_vips):
    try:
        mem_img = img_vips.write_to_memory()
        width, height, bands = img_vips.width, img_vips.height, img_vips.bands

        # Determine bytes per pixel channel based on format
        bytes_per_channel = 1 if img_vips.format == 'uchar' else 2 if img_vips.format == 'ushort' else None
        if bytes_per_channel is None:
            print(f"Unsupported image format: {img_vips.format}")
            return None

        expected_size = width * height * bands * bytes_per_channel
        np_img = np.frombuffer(mem_img, dtype=(np.uint8 if bytes_per_channel == 1 else np.uint16))

        if np_img.size * np_img.itemsize == expected_size:
            np_img = np_img.reshape(height, width, bands)
            if bands == 1:
                np_img = np.repeat(np_img, 3, axis=2)  # Convert grayscale to RGB
            elif bands == 4:
                np_img = np_img[:, :, :3]  # Discard alpha channel

            if bytes_per_channel == 2:
                np_img = (np_img / 256).astype(np.uint8)

            return Image.fromarray(np_img)
        else:
            print(f"Size mismatch for {img_vips.filename}. Expected: {expected_size}, Got: {np_img.size * np_img.itemsize}")
            return None
    except Exception as e:
        print(f"Error converting {img_vips.filename} with pyvips: {e}")
        return None

def process_directory(directory):
    # Supported extensions
    supported_extensions = ('.webp', '.avif', '.heic', '.png')
    all_in_correct_format = True
    found_files = False

    # Traverse each folder and subfolder
    for dirpath, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(supported_extensions):
                found_files = True
                filepath = os.path.join(dirpath, filename)
                try:
                    img = None
                    if filename.lower().endswith('.avif'):
                        print(f"Processing {filename} with pyvips.")
                        img_vips = pyvips.Image.new_from_file(filepath, access='sequential')
                        img = vips_to_pillow(img_vips)
                        if img is None:
                            print(f"Could not convert {filename} with pyvips.")
                            all_in_correct_format = False
                            continue
                    else:
                        img = Image.open(filepath)

                    if img.mode in ("RGBA", "P", "L"):
                        img = img.convert("RGB")

                    jpeg_filepath = os.path.splitext(filepath)[0] + ".jpeg"
                    img.save(jpeg_filepath, "JPEG", quality=95)
                    os.remove(filepath)
                    print(f"Converted and replaced: {filename} -> {os.path.basename(jpeg_filepath)}")
                    all_in_correct_format = False

                except UnidentifiedImageError:
                    print(f"Unidentified image error for {filename}. Skipping.")
                    all_in_correct_format = False
                except Exception as e:
                    print(f"Failed to convert {filename}: {e}")
                    all_in_correct_format = False

    if not found_files:
        print("No files found for conversion.")
    elif all_in_correct_format:
        print("All files are already in the correct format.")
    else:
        print("Conversion ended.")

if __name__ == "__main__":
    print("Conversion started...")
    process_directory(root_directory)
