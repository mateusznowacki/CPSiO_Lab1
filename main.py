import tkinter as tk
from tkinter import filedialog
import numpy as np
import os

# Matplotlib – kluczowe elementy do osadzenia wykresu i toolbaru w tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class PlatformaEKG:
    """
    Klasa umożliwiająca wczytywanie i obsługę sygnałów EKG z plików tekstowych.
    """

    def __init__(self):
        self.sygnaly = None   # Tablica 2D: (liczba_prob, liczba_kanalow)
        self.t = None         # Wektor czasu (jeśli występuje lub sztucznie generowany)
        self.fs = None        # Częstotliwość próbkowania
        self.nazwa_pliku = None

    def wczytaj_plik(self, sciezka_pliku: str):
        self.nazwa_pliku = os.path.basename(sciezka_pliku)
        dane = np.loadtxt(sciezka_pliku)

        # Reset stanu
        self.sygnaly = None
        self.t = None
        self.fs = None

        # Rozpoznawanie formatu
        if self.nazwa_pliku == 'ekg1.txt':
            # 12 kanałów, fs=1000
            self.fs = 1000
            self.sygnaly = dane
            liczba_probek = self.sygnaly.shape[0]
            self.t = np.arange(liczba_probek) / self.fs

        elif self.nazwa_pliku == 'ekg100.txt':
            # 1 kolumna, fs=360
            self.fs = 360
            self.sygnaly = dane.reshape(-1, 1)
            liczba_probek = self.sygnaly.shape[0]
            self.t = np.arange(liczba_probek) / self.fs

        elif self.nazwa_pliku == 'ekg_noise.txt':
            # 2 kolumny: [czas, amplituda], fs=360
            self.fs = 360
            self.t = dane[:, 0]
            self.sygnaly = dane[:, 1].reshape(-1, 1)

        else:
            # Automatyczne rozpoznawanie
            kolumny = dane.shape[1]
            if kolumny == 1:
                # fs=360
                self.fs = 360
                self.sygnaly = dane.reshape(-1, 1)
                liczba_probek = self.sygnaly.shape[0]
                self.t = np.arange(liczba_probek) / self.fs
            elif kolumny == 2:
                # [czas, sygnał]
                self.fs = 360
                self.t = dane[:, 0]
                self.sygnaly = dane[:, 1].reshape(-1, 1)
            else:
                # Wiele kanałów -> fs=1000
                self.fs = 1000
                self.sygnaly = dane
                liczba_probek = self.sygnaly.shape[0]
                self.t = np.arange(liczba_probek) / self.fs

        print(f"Wczytano plik: {self.nazwa_pliku}")
        if self.sygnaly is not None:
            print(f"Kształt sygnału: {self.sygnaly.shape}, fs={self.fs} Hz")

    def pobierz_calosc(self):
        """Zwraca (t, sygnaly) – całość danych."""
        return self.t, self.sygnaly

    def zapisz_fragment_do_pliku(self, czas_start: float, czas_koniec: float, sciezka_wyj: str):
        """
        Zapisuje wycinek sygnału w [czas_start, czas_koniec] do pliku.
        """
        if self.sygnaly is None or self.t is None:
            print("Brak wczytanego sygnału!")
            return

        if czas_start < 0:
            czas_start = 0.0
        if czas_koniec <= czas_start:
            print("Błędny zakres czasu do zapisu!")
            return

        idx_start = np.searchsorted(self.t, czas_start)
        idx_koniec = np.searchsorted(self.t, czas_koniec)
        idx_start = max(idx_start, 0)
        idx_koniec = min(idx_koniec, len(self.t))

        if idx_start >= idx_koniec:
            print("Przedział czasu wykracza poza dane.")
            return

        t_fragment = self.t[idx_start:idx_koniec]
        sygnal_fragment = self.sygnaly[idx_start:idx_koniec, :]

        # Montujemy do jednej macierzy: [czas, amplitudy...]
        dane_do_zapisu = np.column_stack((t_fragment, sygnal_fragment))
        np.savetxt(sciezka_wyj, dane_do_zapisu, fmt='%.6f')
        print(f"Zapisano fragment do pliku: {sciezka_wyj}")

class EKGApp(tk.Tk):
    """
    Aplikacja z wykresem (matplotlib) i standardowym toolbar'em (pan, zoom, home).
    """
    def __init__(self):
        super().__init__()
        self.title("Platforma EKG - interaktywny pan & zoom")
        self.geometry("1000x700")

        self.platforma = PlatformaEKG()

        # Ramka na górne przyciski
        self.top_frame = tk.Frame(self)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Ramka na wykres (dolna)
        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Stworzenie obiektu Figure
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Wykres EKG")
        self.ax.set_xlabel("Czas [s]")
        self.ax.set_ylabel("Amplituda")

        # Umieszczenie canvasa w ramce
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Dodanie paska narzędzi (NavigationToolbar2Tk)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        # Pola wejściowe do zapisu fragmentu
        tk.Label(self.top_frame, text="Plik wyjściowy:").pack(side=tk.LEFT, padx=5)
        self.entry_outfile = tk.Entry(self.top_frame, width=20)
        self.entry_outfile.insert(0, "fragment.txt")
        self.entry_outfile.pack(side=tk.LEFT, padx=5)

        tk.Label(self.top_frame, text="Zakres (start, end) [s]:").pack(side=tk.LEFT, padx=5)
        self.entry_start = tk.Entry(self.top_frame, width=5)
        self.entry_start.insert(0, "0.0")
        self.entry_start.pack(side=tk.LEFT)
        self.entry_end = tk.Entry(self.top_frame, width=5)
        self.entry_end.insert(0, "1.0")
        self.entry_end.pack(side=tk.LEFT)

        btn_save = tk.Button(self.top_frame, text="Zapisz fragment", command=self._on_save_fragment)
        btn_save.pack(side=tk.LEFT, padx=5)

        btn_load = tk.Button(self.top_frame, text="Wczytaj plik EKG", command=self._on_load_file)
        btn_load.pack(side=tk.LEFT, padx=5)

    def _on_load_file(self):
        """Wczytanie pliku i narysowanie przebiegu na wykresie."""
        file_path = filedialog.askopenfilename(
            title="Wybierz plik EKG",
            filetypes=[("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*")]
        )
        if file_path:
            self.platforma.wczytaj_plik(file_path)
            self._narysuj_caly_sygnal()

    def _narysuj_caly_sygnal(self):
        """Rysuje całą dostępną długość sygnału (lub sygnałów) w osi czasu."""
        t, sygnaly = self.platforma.pobierz_calosc()
        self.ax.clear()

        if t is not None and sygnaly is not None:
            if sygnaly.shape[1] == 1:
                # Pojedynczy kanał
                self.ax.plot(t, sygnaly[:, 0], label="EKG")
            else:
                # Wiele kanałów
                for i in range(sygnaly.shape[1]):
                    self.ax.plot(t, sygnaly[:, i], label=f"Ch {i+1}")
                self.ax.legend()

        self.ax.set_title("Wykres EKG")
        self.ax.set_xlabel("Czas [s]")
        self.ax.set_ylabel("Amplituda")
        self.canvas.draw()

    def _on_save_fragment(self):
        """Zapisuje fragment (start, end) do pliku."""
        start_txt = self.entry_start.get().strip()
        end_txt = self.entry_end.get().strip()
        out_file = self.entry_outfile.get().strip()

        if not out_file:
            print("Podaj nazwę pliku wyjściowego!")
            return

        try:
            czas_start = float(start_txt)
            czas_end = float(end_txt)
        except ValueError:
            print("Błędny format czasu!")
            return

        # Zapis do pliku
        sciezka_wyj = os.path.join(os.getcwd(), out_file)
        self.platforma.zapisz_fragment_do_pliku(czas_start, czas_end, sciezka_wyj)

def main():
    app = EKGApp()
    app.mainloop()

if __name__ == "__main__":
    main()
