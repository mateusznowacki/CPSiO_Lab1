import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

def ensure_output_dir(folder="transformed"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def show_image_comparison(original, processed, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Oryginalny")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def local_histogram_equalization(img_array, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_array)

def local_statistics_enhancement(img_array, window_size=15, k=0.5):
    """
    Poprawa jakości na podstawie lokalnych statystyk.
    Wzór: s(x,y) = m(x,y) + k * (r(x,y) - m(x,y)) / std(x,y)
    """
    # Zamiana na float do dokładnych obliczeń
    img = img_array.astype(np.float64)
    kernel = (window_size, window_size)

    # lokalna średnia
    local_mean = cv2.blur(img, kernel)

    # lokalne odchylenie standardowe
    local_sqr = cv2.blur(img**2, kernel)
    local_std = np.sqrt(local_sqr - local_mean**2 + 1e-8)

    # wzór poprawy jakości
    enhanced = local_mean + k * (img - local_mean) / (local_std + 1e-8)
    enhanced = np.clip(enhanced, 0, 255)
    return enhanced.astype(np.uint8)

if __name__ == "__main__":
    files_dir = "files"
    output_dir = ensure_output_dir("transformed")

    filename = "hidden-symbols.tif"
    path = os.path.join(files_dir, filename)
    img = Image.open(path).convert("L")
    arr = np.array(img)

    ### A) LOKALNE WYRÓWNYWANIE HISTOGRAMU ###
    for size in [8, 16, 32]:
        result = local_histogram_equalization(arr, tile_grid_size=(size, size))
        title = f"CLAHE {size}x{size}"
        show_image_comparison(arr, result, title)
        out_name = filename.replace(".tif", f"_clahe_{size}x{size}.tif")
        Image.fromarray(result).save(os.path.join(output_dir, out_name))

    ### B) POPRAWA NA PODSTAWIE LOKALNYCH STATYSTYK ###
    for size in [15, 31, 61]:
        result = local_statistics_enhancement(arr, window_size=size, k=0.8)
        title = f"Lokalna statystyka {size}x{size}"
        show_image_comparison(arr, result, title)
        out_name = filename.replace(".tif", f"_localstats_{size}x{size}.tif")
        Image.fromarray(result).save(os.path.join(output_dir, out_name))
