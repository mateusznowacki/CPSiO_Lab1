import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def multiply_constant(img_array, c):
    """
    Mnożenie obrazu przez stałą:
    T(r) = c * r
    """
    out = c * img_array
    # Obcinamy do zakresu 0..255 i konwertujemy do typu uint8
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

def logarithmic_transform(img_array):
    """
    Transformacja logarytmiczna:
    T(r) = c * log(1 + r)
    gdzie c = 255 / log(256), aby wynik mieścił się w 0..255
    """
    c = 255.0 / np.log(256.0)
    out = c * np.log(1.0 + img_array.astype(np.float64))
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

def contrast_transform(img_array, m=0.45, e=8):
    """
    Zmiana dynamiki skali szarości (kontrastu):
    T(r) = 1 / [1 + (m/r)^e ]
    Przykładowe parametry: m=0.45, e=8
    """
    epsilon = 1e-10
    arr_float = img_array.astype(np.float64)
    # Unikamy dzielenia przez zero
    arr_float[arr_float == 0] = epsilon
    nr = arr_float / 255.0
    out = 1.0 / (1.0 + (m / nr)**e)
    out = out * 255.0
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

def gamma_correction(img_array, c, gamma):
    """
    Korekcja gamma:
    s = c * r^gamma
    Zwykle r jest unormowane do [0..1].
    """
    nr = img_array.astype(np.float64) / 255.0
    out = c * (nr ** gamma)
    out = out * 255.0
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

def plot_transform_function(m=0.45, e=8):
    """
    Wykres T(r) = 1 / [1 + (m/r)^e ], gdzie r z zakresu 0..1.
    Służy do zobrazowania wpływu parametrów m i e na kształt transformacji.
    """
    rs = np.linspace(0.001, 1, 500)
    T = 1.0 / (1.0 + (m / rs)**e)
    plt.figure()
    plt.title(f"Funkcja T(r) = 1 / [1 + (m/r)^e ], m={m}, e={e}")
    plt.xlabel("r (unormowane 0..1)")
    plt.ylabel("T(r)")
    plt.plot(rs, T)
    plt.show()

if __name__ == "__main__":
    # -- UWAGA --
    # Upewnij się, że folder 'files' z obrazami istnieje
    # w tej samej lokalizacji co plik z tym skryptem.
    # Nazwy plików muszą być dokładnie takie, jak użyte w kodzie.

    # A) Mnożenie przez stałą
    chest_file = "chest-xray.tif"  # np. "chest_xray.tif" lub "chest-xray.tif" w zależności od nazwy
    chest_path = os.path.join("files", chest_file)
    chest_img = Image.open(chest_path).convert("L")
    chest_arr = np.array(chest_img)

    c_mult = 1.5
    chest_multiplied = multiply_constant(chest_arr, c_mult)
    # Zapis i wyświetlenie
    multiplied_name = chest_file.replace(".tif", f"_multiplied_{c_mult:.2f}.tif")
    Image.fromarray(chest_multiplied).save(os.path.join("files", multiplied_name))
    # Podgląd w zależności od potrzeb:
    # Image.fromarray(chest_multiplied).show()

    pollen_dark_file = "pollen-dark.tif"
    pollen_dark_path = os.path.join("files", pollen_dark_file)
    pollen_dark_img = Image.open(pollen_dark_path).convert("L")
    pollen_dark_arr = np.array(pollen_dark_img)

    c_mult = 0.5
    pollen_multiplied = multiply_constant(pollen_dark_arr, c_mult)
    multiplied_name = pollen_dark_file.replace(".tif", f"_multiplied_{c_mult:.2f}.tif")
    Image.fromarray(pollen_multiplied).save(os.path.join("files", multiplied_name))

    spectrum_file = "spectrum.tif"
    spectrum_path = os.path.join("files", spectrum_file)
    spectrum_img = Image.open(spectrum_path).convert("L")
    spectrum_arr = np.array(spectrum_img)

    c_mult = 2.0
    spectrum_multiplied = multiply_constant(spectrum_arr, c_mult)
    multiplied_name = spectrum_file.replace(".tif", f"_multiplied_{c_mult:.2f}.tif")
    Image.fromarray(spectrum_multiplied).save(os.path.join("files", multiplied_name))

    # B) Transformacja logarytmiczna
    spectrum_log = logarithmic_transform(spectrum_arr)
    log_name = spectrum_file.replace(".tif", "_log.tif")
    Image.fromarray(spectrum_log).save(os.path.join("files", log_name))

    # C) Zmiana dynamiki skali szarości (kontrastu)
    # Możesz to samo zrobić z chest-xray.tif, einstein-low-contrast.tif, pollen-lowcontrast.tif
    einstein_file = "einstein-low-contrast.tif"
    einstein_path = os.path.join("files", einstein_file)
    einstein_img = Image.open(einstein_path).convert("L")
    einstein_arr = np.array(einstein_img)

    # Parametry m, e
    m_param = 0.45
    e_param = 8
    einstein_contrast = contrast_transform(einstein_arr, m=m_param, e=e_param)
    contrast_name = einstein_file.replace(".tif", f"_contrast_m{m_param:.2f}_e{e_param}.tif")
    Image.fromarray(einstein_contrast).save(os.path.join("files", contrast_name))

    pollen_low_file = "pollen-lowcontrast.tif"
    pollen_low_path = os.path.join("files", pollen_low_file)
    pollen_low_img = Image.open(pollen_low_path).convert("L")
    pollen_low_arr = np.array(pollen_low_img)
    pollen_contrast = contrast_transform(pollen_low_arr, m=m_param, e=e_param)
    contrast_name = pollen_low_file.replace(".tif", f"_contrast_m{m_param:.2f}_e{e_param}.tif")
    Image.fromarray(pollen_contrast).save(os.path.join("files", contrast_name))

    # Możesz też spróbować tej samej transformacji na chest-xray.tif:
    # chest_contrast = contrast_transform(chest_arr, m=m_param, e=e_param)
    # contrast_name = chest_file.replace(".tif", f"_contrast_m{m_param:.2f}_e{e_param}.tif")
    # Image.fromarray(chest_contrast).save(os.path.join("files", contrast_name))

    # Wykres T(r) = 1 / [1 + (m/r)^e ]
    plot_transform_function(m=m_param, e=e_param)

    # D) Korekcja gamma
    aerial_file = "aerial_view.tif"
    aerial_path = os.path.join("files", aerial_file)
    aerial_img = Image.open(aerial_path).convert("L")
    aerial_arr = np.array(aerial_img)
    c_gamma = 1.0
    gamma_value = 2.2
    aerial_gamma = gamma_correction(aerial_arr, c=c_gamma, gamma=gamma_value)
    gamma_name = aerial_file.replace(".tif", f"_gamma_{gamma_value:.2f}.tif")
    Image.fromarray(aerial_gamma).save(os.path.join("files", gamma_name))

    # Koniec – wszystkie obrazy przetworzone zostaną zapisane w folderze 'files'.
