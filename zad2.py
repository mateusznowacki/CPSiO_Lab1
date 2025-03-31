import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import matplotlib.pyplot as plt

class FFTAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analiza FFT (4 wykresy) + Pan/Zoom + wpisywanie zakresów dla sumy i widma")
        self.geometry("1400x800")

        # Ramka z lewej (kontrolki) + ramka z prawej (plot)
        self.left_frame = tk.Frame(self)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Przycisk: wykonaj analizę ---
        btn_run = tk.Button(self.left_frame, text="Uruchom analizę", command=self.run_analysis)
        btn_run.pack(pady=5)

        # --- POLA TEKSTOWE i PRZYCISK do GÓRNEGO LEWEGO (sin(50Hz), czas) ---
        tk.Label(self.left_frame, text="[Górny lewy: sin(50Hz), czas]").pack(pady=5)

        tk.Label(self.left_frame, text="X min:").pack()
        self.entry_tl_xmin = tk.Entry(self.left_frame, width=8)
        self.entry_tl_xmin.insert(0, "0.0")
        self.entry_tl_xmin.pack()

        tk.Label(self.left_frame, text="X max:").pack()
        self.entry_tl_xmax = tk.Entry(self.left_frame, width=8)
        self.entry_tl_xmax.insert(0, "0.5")
        self.entry_tl_xmax.pack()

        tk.Label(self.left_frame, text="Y min:").pack()
        self.entry_tl_ymin = tk.Entry(self.left_frame, width=8)
        self.entry_tl_ymin.insert(0, "-1.2")
        self.entry_tl_ymin.pack()

        tk.Label(self.left_frame, text="Y max:").pack()
        self.entry_tl_ymax = tk.Entry(self.left_frame, width=8)
        self.entry_tl_ymax.insert(0, "1.2")
        self.entry_tl_ymax.pack()

        btn_apply_tl = tk.Button(self.left_frame, text="Ustaw zakres (górny lewy)",
                                 command=self.update_top_left_axes)
        btn_apply_tl.pack(pady=5)

        # --- POLA TEKSTOWE i PRZYCISK do DOLNEGO LEWEGO (suma(50,60), czas) ---
        tk.Label(self.left_frame, text="[Dolny lewy: suma(50,60), czas]").pack(pady=5)

        tk.Label(self.left_frame, text="X min:").pack()
        self.entry_bl_xmin = tk.Entry(self.left_frame, width=8)
        self.entry_bl_xmin.insert(0, "0.0")
        self.entry_bl_xmin.pack()

        tk.Label(self.left_frame, text="X max:").pack()
        self.entry_bl_xmax = tk.Entry(self.left_frame, width=8)
        self.entry_bl_xmax.insert(0, "0.5")
        self.entry_bl_xmax.pack()

        tk.Label(self.left_frame, text="Y min:").pack()
        self.entry_bl_ymin = tk.Entry(self.left_frame, width=8)
        self.entry_bl_ymin.insert(0, "-2.0")
        self.entry_bl_ymin.pack()

        tk.Label(self.left_frame, text="Y max:").pack()
        self.entry_bl_ymax = tk.Entry(self.left_frame, width=8)
        self.entry_bl_ymax.insert(0, "2.0")
        self.entry_bl_ymax.pack()

        btn_apply_bl = tk.Button(self.left_frame, text="Ustaw zakres (dolny lewy)",
                                 command=self.update_bottom_left_axes)
        btn_apply_bl.pack(pady=5)

        # --- POLA TEKSTOWE i PRZYCISK do DOLNEGO PRAWEGO (suma(50,60) – widmo) ---
        tk.Label(self.left_frame, text="[Dolny prawy: suma(50,60), widmo]").pack(pady=5)

        tk.Label(self.left_frame, text="X min:").pack()
        self.entry_br_xmin = tk.Entry(self.left_frame, width=8)
        self.entry_br_xmin.insert(0, "0.0")  # min freq
        self.entry_br_xmin.pack()

        tk.Label(self.left_frame, text="X max:").pack()
        self.entry_br_xmax = tk.Entry(self.left_frame, width=8)
        self.entry_br_xmax.insert(0, "200.0")  # np. do 200 Hz
        self.entry_br_xmax.pack()

        tk.Label(self.left_frame, text="Y min:").pack()
        self.entry_br_ymin = tk.Entry(self.left_frame, width=8)
        self.entry_br_ymin.insert(0, "0.0")
        self.entry_br_ymin.pack()

        tk.Label(self.left_frame, text="Y max:").pack()
        self.entry_br_ymax = tk.Entry(self.left_frame, width=8)
        self.entry_br_ymax.insert(0, "1100.0")  # np. amplitude
        self.entry_br_ymax.pack()

        btn_apply_br = tk.Button(self.left_frame, text="Ustaw zakres (dolny prawy)",
                                 command=self.update_bottom_right_axes)
        btn_apply_br.pack(pady=5)

        # Stworzenie figury 2x2 subploty
        self.fig, self.axs = plt.subplots(2, 2, figsize=(8, 6))
        self.fig.tight_layout(pad=4.0)

        # Osadzenie figure w tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Pasek narzędzi (Pan/Zoom)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def run_analysis(self):
        """
        Generujemy:
          - sin(50 Hz) [Górny wiersz],
          - sumę sin(50 Hz + 60 Hz) [Dolny wiersz].
        Obliczamy ich widma i rysujemy w 2x2 subplotach:
          Górny lewy - sin(50), czas
          Górny prawy - widmo sin(50)
          Dolny lewy - suma(50,60), czas
          Dolny prawy - widmo sumy(50,60).
        """
        fs = 1000
        N = 65536
        t = np.arange(N) / fs

        # --- GÓRNY WIERSZ: sin(50 Hz) ---
        freq1 = 50
        y_sin = np.sin(2*np.pi*freq1*t)

        fft_sin = np.fft.fft(y_sin)
        amp_sin = np.abs(fft_sin)
        freqs = np.fft.fftfreq(N, d=1/fs)
        half = N // 2

        ax_tl = self.axs[0][0]
        ax_tr = self.axs[0][1]

        ax_tl.clear()
        ax_tl.plot(t, y_sin, label="sin(50 Hz)")
        ax_tl.set_title("sin(50 Hz) - czas", fontsize=10)
        ax_tl.set_xlabel("Czas [s]", fontsize=9)
        ax_tl.set_ylabel("Amplituda", fontsize=9)
        ax_tl.legend()
        ax_tl.grid(True)

        ax_tr.clear()
        ax_tr.plot(freqs[:half], amp_sin[:half], label="Widmo sin(50 Hz)")
        ax_tr.set_title("Widmo sin(50 Hz)", fontsize=10)
        ax_tr.set_xlabel("Częstotliwość [Hz]", fontsize=9)
        ax_tr.set_ylabel("Amplituda", fontsize=9)
        ax_tr.legend()
        ax_tr.grid(True)

        # --- DOLNY WIERSZ: suma(50 Hz + 60 Hz) ---
        freq2 = 60
        y_mix = np.sin(2*np.pi*freq1*t) + np.sin(2*np.pi*freq2*t)

        fft_mix = np.fft.fft(y_mix)
        amp_mix = np.abs(fft_mix)

        ax_bl = self.axs[1][0]
        ax_br = self.axs[1][1]

        ax_bl.clear()
        ax_bl.plot(t, y_mix, label="sum(50,60)")
        ax_bl.set_title("Suma 50 i 60 Hz - czas", fontsize=10)
        ax_bl.set_xlabel("Czas [s]", fontsize=9)
        ax_bl.set_ylabel("Amplituda", fontsize=9)
        ax_bl.legend()
        ax_bl.grid(True)

        ax_br.clear()
        ax_br.plot(freqs[:half], amp_mix[:half], label="Widmo sumy(50+60)")
        ax_br.set_title("Widmo sumy (50 + 60 Hz)", fontsize=10)
        ax_br.set_xlabel("Częstotliwość [Hz]", fontsize=9)
        ax_br.set_ylabel("Amplituda", fontsize=9)
        ax_br.legend()
        ax_br.grid(True)

        self.canvas.draw()

    def update_top_left_axes(self):
        """
        Odczytujemy pola [X min, X max, Y min, Y max] dla górnego lewego subplotu
        (sin(50 Hz) w dziedzinie czasu).
        """
        ax = self.axs[0][0]

        try:
            x_min = float(self.entry_tl_xmin.get())
            x_max = float(self.entry_tl_xmax.get())
            y_min = float(self.entry_tl_ymin.get())
            y_max = float(self.entry_tl_ymax.get())
        except ValueError:
            print("Błędny format wartości (górny lewy)!")
            return

        if x_min < x_max:
            ax.set_xlim(x_min, x_max)
        if y_min < y_max:
            ax.set_ylim(y_min, y_max)

        self.canvas.draw()

    def update_bottom_left_axes(self):
        """
        Odczytujemy pola [X min, X max, Y min, Y max] dla dolnego lewego subplotu
        (suma(50,60) w dziedzinie czasu).
        """
        ax = self.axs[1][0]

        try:
            x_min = float(self.entry_bl_xmin.get())
            x_max = float(self.entry_bl_xmax.get())
            y_min = float(self.entry_bl_ymin.get())
            y_max = float(self.entry_bl_ymax.get())
        except ValueError:
            print("Błędny format wartości (dolny lewy)!")
            return

        if x_min < x_max:
            ax.set_xlim(x_min, x_max)
        if y_min < y_max:
            ax.set_ylim(y_min, y_max)

        self.canvas.draw()

    def update_bottom_right_axes(self):
        """
        Odczytujemy pola [X min, X max, Y min, Y max] dla dolnego prawego subplotu
        (widmo sumy(50,60)).
        """
        ax = self.axs[1][1]

        try:
            x_min = float(self.entry_br_xmin.get())
            x_max = float(self.entry_br_xmax.get())
            y_min = float(self.entry_br_ymin.get())
            y_max = float(self.entry_br_ymax.get())
        except ValueError:
            print("Błędny format wartości (dolny prawy)!")
            return

        if x_min < x_max:
            ax.set_xlim(x_min, x_max)
        if y_min < y_max:
            ax.set_ylim(y_min, y_max)

        self.canvas.draw()


def main():
    app = FFTAnalysisApp()
    app.mainloop()

if __name__ == "__main__":
    main()
