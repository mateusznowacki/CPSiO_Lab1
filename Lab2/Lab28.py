import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.util import img_as_float
from skimage.filters import (
    gaussian,
    median,
    laplace,
    unsharp_mask
)
from skimage.morphology import disk
from skimage.exposure import rescale_intensity


def main():
    # 1) Wczytanie obrazu z podfolderu 'files'
    image_path = os.path.join("files", "bonescan.tif")
    image = io.imread(image_path)

    # 2) Konwersja do float [0..1]
    image = img_as_float(image)

    # Prosty podgląd oryginału (opcjonalnie)
    plt.figure("Oryginał")
    plt.imshow(image, cmap='gray')
    plt.title("Oryginalny obraz (bonescan)")
    plt.axis('off')

    # 3) Filtr Gaussa (usuwa drobne zakłócenia HF)
    gauss_sigma = 2.0
    image_gauss = gaussian(image, sigma=gauss_sigma)

    # 4) Filtr medianowy – usuwa szum impulsowy
    #    Zwróć uwagę na użycie 'footprint=' zamiast 'selem='
    image_med = median(image_gauss, footprint=disk(3))

    # 5) Wyostrzanie - Laplace
    lap = laplace(image_med, ksize=3)
    sharpen_lap = image_med - 0.2 * lap  # waga 0.2 -> doświadczalna

    # 6) Reskalowanie intensywności
    sharpen_lap_rescale = rescale_intensity(sharpen_lap, in_range='image', out_range=(0, 1))

    # 7) Unsharp masking
    unsharp = unsharp_mask(image_med, radius=2.0, amount=1.5)
    unsharp_rescale = rescale_intensity(unsharp, in_range='image', out_range=(0, 1))

    # 8) Wizualizacja w siatce 2x3
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax = axes.ravel()

    # Oryginał
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Oryginał")
    ax[0].axis('off')

    # Po Gaussie i Medianie
    ax[1].imshow(image_med, cmap='gray')
    ax[1].set_title("Po Gaussie i Medianie")
    ax[1].axis('off')

    # Wyostrzony Laplace + rescale
    ax[2].imshow(sharpen_lap_rescale, cmap='gray')
    ax[2].set_title("Wyostrzony (Laplace)")
    ax[2].axis('off')

    # Sama mapa Laplasjanu
    ax[3].imshow(lap, cmap='gray')
    ax[3].set_title("Mapa Laplasjanu")
    ax[3].axis('off')

    # Unsharp mask
    ax[4].imshow(unsharp, cmap='gray')
    ax[4].set_title("Unsharp masking")
    ax[4].axis('off')

    # Unsharp + rescale
    ax[5].imshow(unsharp_rescale, cmap='gray')
    ax[5].set_title("Unsharp + rescale_intensity")
    ax[5].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
