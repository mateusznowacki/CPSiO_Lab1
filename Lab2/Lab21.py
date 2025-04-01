import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np


class ImageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Viewer")

        # Kontrolki GUI
        self.open_button = tk.Button(self, text="Otwórz obraz", command=self.open_image)
        self.open_button.pack(pady=5)

        self.line_profile_button_h = tk.Button(self, text="Profil poziomy", command=self.plot_horizontal_profile)
        self.line_profile_button_h.pack(pady=5)

        self.line_profile_button_v = tk.Button(self, text="Profil pionowy", command=self.plot_vertical_profile)
        self.line_profile_button_v.pack(pady=5)

        self.crop_button = tk.Button(self, text="Wytnij podobszar", command=self.crop_subimage)
        self.crop_button.pack(pady=5)

        self.save_crop_button = tk.Button(self, text="Zapisz podobszar", command=self.save_subimage)
        self.save_crop_button.pack(pady=5)

        self.label = tk.Label(self)
        self.label.pack(padx=5, pady=5)

        # Zmienne przechowujące obraz
        self.original_image = None  # PIL.Image
        self.photo_image = None  # ImageTk.PhotoImage
        self.np_image = None  # numpy array
        self.cropped_image = None  # PIL.Image do zapisu

    def open_image(self):
        """Wczytywanie obrazu z pliku i wyświetlanie w GUI."""
        filename = filedialog.askopenfilename(
            title="Wybierz plik z obrazem",
            filetypes=[
                ("Obrazy TIFF", "*.tif *.tiff"),
                ("Obrazy PNG", "*.png"),
                ("Obrazy JPEG", "*.jpg *.jpeg"),
                ("Wszystkie pliki", "*.*")
            ]
        )
        if filename:
            self.original_image = Image.open(filename)
            self.np_image = np.array(self.original_image)

            self.show_image(self.original_image)

    def show_image(self, img):
        """Wyświetla PIL Image w kontrolce Label."""
        self.photo_image = ImageTk.PhotoImage(img)
        self.label.config(image=self.photo_image)
        self.label.image = self.photo_image  # zapobiega usunięciu obiektu z pamięci

    def plot_horizontal_profile(self):
        """Prosi użytkownika o podanie współrzędnej wiersza i rysuje profil poziomy."""
        if self.np_image is None:
            return

        row = simpledialog.askinteger("Profil poziomy", f"Podaj wiersz (0 <= wysokość <= {self.np_image.shape[0] - 1}):")
        if row is None:
            return

        if row < 0 or row >= self.np_image.shape[0]:
            messagebox.showerror("Błąd", "Nieprawidłowa współrzędna wiersza!")
            return

        # Jeśli obraz jest w skali szarości (2 wymiary)
        if len(self.np_image.shape) == 2:
            line_data = self.np_image[row, :]
        else:
            # Kolorowy: robimy średnią wzdłuż kanałów
            line_data = np.mean(self.np_image[row, :, :], axis=-1)

        # Rysunek w oknie matplotlib
        plt.figure("Profil poziomy")
        plt.clf()
        plt.title(f"Profil poziomy w wierszu {row}")
        plt.plot(line_data, color='black')
        plt.xlabel("Kolumna")
        plt.ylabel("Wartość piksela")
        plt.show()

    def plot_vertical_profile(self):
        """Prosi użytkownika o podanie współrzędnej kolumny i rysuje profil pionowy.
           Zmieniono, aby na osi X wyświetlać wiersze, a na osi Y wartości pikseli.
        """
        if self.np_image is None:
            return

        col = simpledialog.askinteger("Profil pionowy", f"Podaj wiersz (0 <= wysokość <= {self.np_image.shape[1] - 1}):")
        if col is None:
            return

        if col < 0 or col >= self.np_image.shape[1]:
            messagebox.showerror("Błąd", "Nieprawidłowa współrzędna kolumny!")
            return

        # Skalę szarości lub średnia w kolumnie (jeśli kolor)
        if len(self.np_image.shape) == 2:
            line_data = self.np_image[:, col]
        else:
            line_data = np.mean(self.np_image[:, col, :], axis=-1)

        # Tutaj zamieniamy osie: wiersz to oś X, a wartość piksela to oś Y
        rows = np.arange(len(line_data))

        plt.figure("Profil pionowy")
        plt.clf()
        plt.title(f"Profil pionowy w kolumnie {col}")
        plt.plot(rows, line_data, color='black')
        plt.xlabel("Wiersz")
        plt.ylabel("Wartość piksela")
        plt.show()

    def crop_subimage(self):
        """Tworzy okno do wpisania współrzędnych prostokąta i wycina podobszar."""
        if self.original_image is None:
            return

        def confirm_crop():
            try:
                x1 = int(entry_x1.get())
                y1 = int(entry_y1.get())
                x2 = int(entry_x2.get())
                y2 = int(entry_y2.get())

                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                if (x1 < 0 or y1 < 0 or x2 > self.original_image.width or y2 > self.original_image.height):
                    messagebox.showerror("Błąd", "Współrzędne wykraczają poza rozmiar obrazu.", parent=popup)
                    return

                # Wycięcie fragmentu i zamknięcie okna
                self.cropped_image = self.original_image.crop((x1, y1, x2, y2))
                self.show_image(self.cropped_image)
                popup.destroy()

            except ValueError:
                messagebox.showerror("Błąd", "Podaj poprawne liczby całkowite.", parent=popup)

        # Okno dialogowe
        popup = tk.Toplevel(self)
        popup.title("Wprowadź współrzędne")
        popup.grab_set()  # Zablokowanie innych okien do czasu zamknięcia

        tk.Label(popup, text="x1 (lewy górny róg):").grid(row=0, column=0, sticky="e")
        tk.Label(popup, text="y1 (lewy górny róg):").grid(row=1, column=0, sticky="e")
        tk.Label(popup, text="x2 (prawy dolny róg):").grid(row=2, column=0, sticky="e")
        tk.Label(popup, text="y2 (prawy dolny róg):").grid(row=3, column=0, sticky="e")

        entry_x1 = tk.Entry(popup)
        entry_y1 = tk.Entry(popup)
        entry_x2 = tk.Entry(popup)
        entry_y2 = tk.Entry(popup)

        entry_x1.grid(row=0, column=1, padx=5, pady=2)
        entry_y1.grid(row=1, column=1, padx=5, pady=2)
        entry_x2.grid(row=2, column=1, padx=5, pady=2)
        entry_y2.grid(row=3, column=1, padx=5, pady=2)

        tk.Button(popup, text="OK", command=confirm_crop).grid(row=4, column=0, columnspan=2, pady=10)

    def save_subimage(self):
        """Zapisuje aktualnie wycięty fragment do wybranego pliku."""
        if self.cropped_image is None:
            messagebox.showinfo("Informacja", "Brak wyciętego fragmentu do zapisania.")
            return

        filename = filedialog.asksaveasfilename(
            title="Zapisz podobszar",
            defaultextension=".tif",
            filetypes=[
                ("TIFF", "*.tif"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("Wszystkie pliki", "*.*")
            ]
        )
        if filename:
            self.cropped_image.save(filename)
            messagebox.showinfo("Informacja", f"Zapisano podobszar do pliku:\n{filename}")


if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()
