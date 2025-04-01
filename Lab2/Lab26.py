import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

def ensure_output_dir(folder="transformed"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def show_comparison(original, filtered, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Oryginalny")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(filtered, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def apply_lowpass_filters(image_array, mask_sizes, filename, output_dir):
    for k in mask_sizes:
        # a) filtr uśredniający (mean)
        avg = cv2.blur(image_array, (k, k))
        title = f"Średnia {k}x{k}"
        show_comparison(image_array, avg, title)
        outname = filename.replace(".tif", f"_mean_{k}x{k}.tif")
        Image.fromarray(avg).save(os.path.join(output_dir, outname))

        # b) filtr Gaussowski
        sigma = 0  # automatyczne dopasowanie
        gauss = cv2.GaussianBlur(image_array, (k, k), sigma)
        title = f"Gaussowski {k}x{k}"
        show_comparison(image_array, gauss, title)
        outname = filename.replace(".tif", f"_gauss_{k}x{k}.tif")
        Image.fromarray(gauss).save(os.path.join(output_dir, outname))

if __name__ == "__main__":
    files_dir = "files"
    output_dir = ensure_output_dir("transformed")

    image_files = [
        "characters_test_pattern.tif",
        "zoneplate.tif"
    ]

    mask_sizes = [3, 7, 15]  # można dodać więcej, np. 21, 31...

    for filename in image_files:
        path = os.path.join(files_dir, filename)
        img = Image.open(path).convert("L")
        arr = np.array(img)

        apply_lowpass_filters(arr, mask_sizes, filename, output_dir)
