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

def apply_all_filters(image_array, mask_sizes, filename, output_dir):
    for k in mask_sizes:
        ### (a) Filtr uśredniający
        avg = cv2.blur(image_array, (k, k))
        title = f"Średnia {k}x{k}"
        show_comparison(image_array, avg, title)
        out = filename.replace(".tif", f"_avg_{k}x{k}.tif")
        Image.fromarray(avg).save(os.path.join(output_dir, out))

        ### (b) Filtr medianowy
        if k % 2 == 1:  # medianBlur wymaga nieparzystego rozmiaru
            med = cv2.medianBlur(image_array, k)
            title = f"Mediana {k}x{k}"
            show_comparison(image_array, med, title)
            out = filename.replace(".tif", f"_median_{k}x{k}.tif")
            Image.fromarray(med).save(os.path.join(output_dir, out))

        ### (c1) Filtr minimum (erode)
        minf = cv2.erode(image_array, np.ones((k, k), np.uint8))
        title = f"Minimum {k}x{k}"
        show_comparison(image_array, minf, title)
        out = filename.replace(".tif", f"_min_{k}x{k}.tif")
        Image.fromarray(minf).save(os.path.join(output_dir, out))

        ### (c2) Filtr maksimum (dilate)
        maxf = cv2.dilate(image_array, np.ones((k, k), np.uint8))
        title = f"Maksimum {k}x{k}"
        show_comparison(image_array, maxf, title)
        out = filename.replace(".tif", f"_max_{k}x{k}.tif")
        Image.fromarray(maxf).save(os.path.join(output_dir, out))

if __name__ == "__main__":
    files_dir = "files"
    output_dir = ensure_output_dir("transformed")

    image_files = [
        "cboard_pepper_only.tif",
        "cboard_salt_only.tif",
        "cboard_salt_pepper.tif"
    ]

    mask_sizes = [3, 5, 7]

    for filename in image_files:
        path = os.path.join(files_dir, filename)
        img = Image.open(path).convert("L")
        arr = np.array(img)

        apply_all_filters(arr, mask_sizes, filename, output_dir)
