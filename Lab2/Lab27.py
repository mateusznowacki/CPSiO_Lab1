import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os


def ensure_output_dir(folder="transformed"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def show(title, image1, image2):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap="gray")
    plt.title("Oryginalny")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def sobel_filters(image_array, filename, output_dir):
    # Krawędzie poziome i pionowe
    sobelx = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.hypot(sobelx, sobely)
    sobel_combined = np.clip(sobel_combined, 0, 255).astype(np.uint8)

    # Zapisy
    Image.fromarray(np.abs(sobelx).astype(np.uint8)).save(os.path.join(output_dir, filename.replace(".", "_sobelx.")))
    Image.fromarray(np.abs(sobely).astype(np.uint8)).save(os.path.join(output_dir, filename.replace(".", "_sobely.")))
    Image.fromarray(sobel_combined).save(os.path.join(output_dir, filename.replace(".", "_sobel_combined.")))

    show("Sobel X + Y (ukośne)", image_array, sobel_combined)


def laplacian_sharpening(image_array, filename, output_dir):
    lap = cv2.Laplacian(image_array, cv2.CV_64F)
    lap_abs = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    sharpened = cv2.add(image_array, lap_abs)

    Image.fromarray(lap_abs).save(os.path.join(output_dir, filename.replace(".", "_laplacian.")))
    Image.fromarray(sharpened).save(os.path.join(output_dir, filename.replace(".", "_laplacian_sharpened.")))

    show("Wyostrzanie Laplasjanem", image_array, sharpened)


def unsharp_and_highboost(image_array, filename, output_dir, k=1.5):
    # Zamieniamy na float, żeby uniknąć ujemnych/za dużych pikseli
    img = image_array.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Unsharp masking = oryginał + (oryg - rozmycie)
    mask = img - blurred
    unsharp = img + mask
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)

    # High-boost = oryginał + k * (oryg - rozmycie)
    highboost = img + k * (img - blurred)
    highboost = np.clip(highboost, 0, 255).astype(np.uint8)

    # Zapis i podgląd
    Image.fromarray(unsharp).save(os.path.join(output_dir, filename.replace(".", "_unsharp.")))
    Image.fromarray(highboost).save(os.path.join(output_dir, filename.replace(".", f"_highboost_k{k}.")))

    show("Unsharp Masking", image_array, unsharp)
    show(f"High-Boost (k={k})", image_array, highboost)


if __name__ == "__main__":
    files_dir = "files"
    output_dir = ensure_output_dir("transformed")

    # a) Sobel – edge detection
    for file in ["circuitmask.tif", "testpat1.png"]:
        path = os.path.join(files_dir, file)
        img = Image.open(path).convert("L")
        arr = np.array(img)
        sobel_filters(arr, file, output_dir)

    # b) Laplacian – sharpening
    file = "blurry-moon.tif"
    path = os.path.join(files_dir, file)
    img = Image.open(path).convert("L")
    arr = np.array(img)
    laplacian_sharpening(arr, file, output_dir)

    # c) Unsharp Masking & High Boost
    file = "text-dipxe-blurred.tif"
    path = os.path.join(files_dir, file)
    img = Image.open(path).convert("L")
    arr = np.array(img)
    unsharp_and_highboost(arr, file, output_dir, k=1.5)
