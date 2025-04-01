import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import matplotlib.pyplot as plt

class ECGFFTApp(tk.Tk):
    def __init__(self, filename, fs):
        super().__init__()
        self.title("Analiza FFT jednego sygnału EKG")
        self.geometry("1400x800")
        self.filename = filename
        self.fs = fs

        # Wczytanie jednego sygnału jako wektora 1D
        self.signal = np.loadtxt(self.filename)
        self.N = len(self.signal)
        self.t = np.arange(self.N) / self.fs
        self.freqs = np.fft.fftfreq(self.N, d=1/self.fs)
        self.half = self.N // 2

        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=4.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        self.bottom_controls = tk.Frame(self)
        self.bottom_controls.pack(side=tk.BOTTOM, pady=10)

        self.run_btn = tk.Button(self.bottom_controls, text="Uruchom analizę", command=self.run_analysis)
        self.run_btn.pack()

        tk.Label(self.bottom_controls, text="Zakres czasu [s]: od").pack(side=tk.LEFT)
        self.start_entry = tk.Entry(self.bottom_controls, width=6)
        self.start_entry.insert(0, "0.0")
        self.start_entry.pack(side=tk.LEFT)

        tk.Label(self.bottom_controls, text="do").pack(side=tk.LEFT)
        self.end_entry = tk.Entry(self.bottom_controls, width=6)
        self.end_entry.insert(0, f"{self.N / self.fs-0.01:.2f}")
        self.end_entry.pack(side=tk.LEFT)

        self.range_btn = tk.Button(self.bottom_controls, text="Zastosuj zakres", command=self.run_analysis)
        self.range_btn.pack(side=tk.LEFT, padx=5)

    def run_analysis(self):
        try:
            time_start = float(self.start_entry.get())
            time_end = float(self.end_entry.get())
        except ValueError:
            print("Zakres czasu musi być liczbą.")
            return

        if time_start < 0 or time_end > self.N / self.fs or time_start >= time_end:
            print("Nieprawidłowy zakres czasu.")
            return

        idx_start = int(time_start * self.fs)
        idx_end = int(time_end * self.fs)

        signal = self.signal[idx_start:idx_end]
        t = np.arange(idx_start, idx_end) / self.fs
        fft_vals = np.fft.fft(signal)
        amplitude = np.abs(fft_vals)
        freqs = np.fft.fftfreq(len(signal), d=1/self.fs)
        half = len(signal) // 2
        reconstructed = np.fft.ifft(fft_vals).real
        difference = signal - reconstructed

        # 1. Oryginalny sygnał
        self.axs[0][0].clear()
        self.axs[0][0].plot(t, signal)
        self.axs[0][0].set_title("Oryginalny sygnał EKG")
        self.axs[0][0].set_xlabel("Czas [s]")
        self.axs[0][0].set_ylabel("Amplituda")
        self.axs[0][0].grid(True)

        # 2. Widmo amplitudowe
        self.axs[0][1].clear()
        self.axs[0][1].plot(freqs[:half], amplitude[:half])
        self.axs[0][1].set_title("Widmo amplitudowe")
        self.axs[0][1].set_xlabel("Częstotliwość [Hz]")
        self.axs[0][1].set_ylabel("Amplituda")
        self.axs[0][1].grid(True)

        # 3. Sygnał po iFFT
        self.axs[1][0].clear()
        self.axs[1][0].plot(t, reconstructed)
        self.axs[1][0].set_title("Sygnał po odwrotnej FFT")
        self.axs[1][0].set_xlabel("Czas [s]")
        self.axs[1][0].set_ylabel("Amplituda")
        self.axs[1][0].grid(True)

        # 4. Różnica
        self.axs[1][1].clear()
        self.axs[1][1].plot(t, difference)
        self.axs[1][1].set_title("Różnica (oryginalny - ifft)")
        self.axs[1][1].set_xlabel("Czas [s]")
        self.axs[1][1].set_ylabel("Amplituda")
        self.axs[1][1].grid(True)

        self.canvas.draw()


def main():
    app = ECGFFTApp("signals/ekg100.txt", fs=360)
    app.mainloop()

if __name__ == "__main__":
    main()
