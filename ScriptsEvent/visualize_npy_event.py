import os
import numpy as np
from PIL import Image


def convert_npy_to_jpeg(root_folder):
    output_folder = os.path.join(root_folder, "output")
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(root_folder):
        if file.endswith(".npy"):
            file_path = os.path.join(root_folder, file)
            img_array = np.load(file_path)  # Load npy file

            if img_array.ndim == 3 and img_array.shape[2] == 2:  # Two-channel image
                # Normalize channels
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

                # Create blue-yellow mapping
                rgb_image = np.zeros((*img_array.shape[:2], 3), dtype=np.uint8)
                rgb_image[..., 0] = (img_array[..., 0] * 255).astype(np.uint8)  # Blue
                rgb_image[..., 1] = ((img_array[..., 0] + img_array[..., 1]) * 127.5).astype(
                    np.uint8)  # Yellow intensity
                rgb_image[..., 2] = (img_array[..., 1] * 255).astype(np.uint8)  # Yellow (mix with red channel)

                img = Image.fromarray(rgb_image)
            else:
                print(f"Skipping {file}: unsupported shape {img_array.shape}")
                continue

            output_path = os.path.join(output_folder, file.replace(".npy", ".jpeg"))
            img.save(output_path, "JPEG")
            print(f"Saved {output_path}")

if __name__ == "__main__":
    root_folder = "C:\\Users\Gianluca\Desktop\EHWGesture\dataset_processed\X05N_R\Prova_TR1\event_TR1"  # Change to your root folder path
    convert_npy_to_jpeg(root_folder)