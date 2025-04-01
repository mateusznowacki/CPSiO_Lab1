import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, filtfilt, freqz
from scipy.fft import fft, fftfreq

class EKGFilterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Filtracja sygnału EKG - Butterworth")
        self.geometry("1200x800")

        self.fig, self.axs = plt.subplots(3, 2, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.load_and_filter_signal()
        self.plot_all()

    def load_and_filter_signal(self):
        # Wczytanie danych
        self.fs = 360
        data = np.loadtxt("signals\\ekg_noise.txt")
        self.t = data[:, 0]
        self.signal = data[:, 1]
        self.N = len(self.signal)
        self.freqs = fftfreq(self.N, d=1/self.fs)

        # FFT oryginalnego sygnału
        self.fft_orig = np.abs(fft(self.signal))

        # Filtr LPF (60 Hz)
        b_low, a_low = butter(4, 60 / (self.fs / 2), btype='low')
        self.filtered_low = filtfilt(b_low, a_low, self.signal)
        self.fft_low = np.abs(fft(self.filtered_low))
        self.diff_low = self.signal - self.filtered_low
        self.fft_diff_low = np.abs(fft(self.diff_low))

        # Filtr HPF (5 Hz)
        b_high, a_high = butter(4, 5 / (self.fs / 2), btype='high')
        self.filtered_final = filtfilt(b_high, a_high, self.filtered_low)
        self.fft_final = np.abs(fft(self.filtered_final))
        self.diff_final = self.signal - self.filtered_final
        self.fft_diff_final = np.abs(fft(self.diff_final))

    def plot_all(self):
        self.axs[0][0].plot(self.t, self.signal)
        self.axs[0][0].set_title("Oryginalny sygnał EKG")

        self.axs[0][1].plot(self.freqs[:self.N // 2], self.fft_orig[:self.N // 2])
        self.axs[0][1].set_title("Widmo oryginalnego sygnału")

        self.axs[1][0].plot(self.t, self.filtered_final)
        self.axs[1][0].set_title("Po filtrach LPF + HPF")

        self.axs[1][1].plot(self.freqs[:self.N // 2], self.fft_final[:self.N // 2])
        self.axs[1][1].set_title("Widmo po filtracji")

        self.axs[2][0].plot(self.t, self.diff_final)
        self.axs[2][0].set_title("Różnica (oryginalny - końcowy)")

        self.axs[2][1].plot(self.freqs[:self.N // 2], self.fft_diff_final[:self.N // 2])
        self.axs[2][1].set_title("Widmo różnicy")

        for ax in self.axs.flat:
            ax.set_xlabel("Czas [s]" if "Widmo" not in ax.get_title() else "Częstotliwość [Hz]")
            ax.set_ylabel("Amplituda")
            ax.grid(True)

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = EKGFilterApp()
    app.mainloop()
