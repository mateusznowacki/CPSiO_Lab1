import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def ensure_output_dir(folder="transformed"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def plot_histograms(original, transformed, title_prefix):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(original.flatten(), bins=256, range=[0, 255], color='gray')
    plt.title(f"{title_prefix} – Histogram przed")

    plt.subplot(1, 2, 2)
    plt.hist(transformed.flatten(), bins=256, range=[0, 255], color='gray')
    plt.title(f"{title_prefix} – Histogram po")
    plt.tight_layout()
    plt.show()

def show_image_comparison(original, transformed, title_prefix):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title(f"{title_prefix} – Oryginalny")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(transformed, cmap="gray")
    plt.title(f"{title_prefix} – Po wyrównaniu")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def equalize_histogram(img_array):
    hist, bins = np.histogram(img_array.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()
    cdf_scaled = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
    cdf_scaled = np.ma.filled(cdf_scaled, 0).astype(np.uint8)
    result = cdf_scaled[img_array]
    return result

if __name__ == "__main__":
    files_dir = "files"
    output_dir = ensure_output_dir("transformed")

    images = [
        "chest-xray.tif",
        "pollen-dark.tif",
        "pollen-ligt.tif",
        "pollen-lowcontrast.tif",
        "pout.tif",
        "spectrum.tif"
    ]

    for filename in images:
        path = os.path.join(files_dir, filename)
        img = Image.open(path).convert("L")
        arr = np.array(img)

        equalized = equalize_histogram(arr)

        # Zapisz do pliku
        output_name = filename.replace(".tif", "_equalized.tif")
        Image.fromarray(equalized).save(os.path.join(output_dir, output_name))

        # Pokaż obrazy oryginalny vs po przekształceniu
        show_image_comparison(arr, equalized, title_prefix=filename)

        # Pokaż histogramy
        plot_histograms(arr, equalized, title_prefix=filename)
