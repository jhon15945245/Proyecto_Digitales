"""Graphical interface for the signal digitization project."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Modules.audio_io import AudioSignal, load_wav, play_audio, stop_audio
from Modules.dsp_core import VALID_LEVELS
from Modules.experiments import run_and_export_quantization, run_and_export_sampling
from Modules.simulation import SimulationResult, simulate_audio, simulate_synthetic


class DigitizationApp:
    """Interactive simulator for sampling, quantization and reconstruction."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Simulación de un sistema de digitalización")
        self.root.geometry("1450x900")
        self.root.minsize(1100, 720)

        self.update_job = None
        self.audio = None
        self.result: SimulationResult | None = None

        self.signal_var = tk.StringVar(value="A")
        self.fs_var = tk.DoubleVar(value=100.0)
        self.fs_entry_var = tk.StringVar(value="100")
        self.zoom_enabled_var = tk.BooleanVar(value=False)
        self.zoom_start_var = tk.StringVar(value="0")
        self.zoom_end_var = tk.StringVar(value="2")
        self.levels_var = tk.StringVar(value="16")
        self.duration_var = tk.DoubleVar(value=2.0)
        self.status_var = tk.StringVar(value="Mueva los controles para actualizar la simulación")
        self.mse_var = tk.StringVar(value="-")
        self.sqnr_var = tk.StringVar(value="-")
        self.spectral_var = tk.StringVar(value="-")
        self.reference_mse_var = tk.StringVar(value="-")
        self.reference_sqnr_var = tk.StringVar(value="-")
        self.reference_spectral_var = tk.StringVar(value="-")
        self.reference_info_var = tk.StringVar(value="Referencia: -")
        self.info_var = tk.StringVar(value="Sin archivo de audio")
        self.step_var = tk.StringVar(value="Δ = -")
        self.ts_var = tk.StringVar(value="Ts = -")

        self._build_controls()
        self._build_plots()
        self._schedule_update()

    def _build_controls(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Señal:").grid(row=0, column=0, sticky="w")
        signal_menu = ttk.Combobox(
            controls,
            textvariable=self.signal_var,
            values=("A", "B", "C", "D - Audio"),
            state="readonly",
            width=12,
        )
        signal_menu.grid(row=0, column=1, padx=(4, 18))
        signal_menu.bind("<<ComboboxSelected>>", self._on_signal_changed)

        ttk.Label(controls, text="Niveles:").grid(row=0, column=2, sticky="w")
        levels_menu = ttk.Combobox(
            controls,
            textvariable=self.levels_var,
            values=VALID_LEVELS,
            state="readonly",
            width=8,
        )
        levels_menu.grid(row=0, column=3, sticky="w", padx=(4, 18))
        levels_menu.bind("<<ComboboxSelected>>", lambda _event: self._schedule_update())

        ttk.Label(controls, text="fs (Hz):").grid(row=0, column=4, sticky="w")
        self.fs_entry = ttk.Entry(controls, textvariable=self.fs_entry_var, width=10)
        self.fs_entry.grid(row=0, column=5, padx=(4, 4))
        self.fs_entry.bind("<Return>", lambda _event: self.apply_frequency())
        ttk.Button(controls, text="Aplicar fs", command=self.apply_frequency).grid(row=0, column=6, padx=(0, 12))

        ttk.Label(controls, text="Duración (s):").grid(row=0, column=7, sticky="w")
        self.duration_scale = tk.Scale(
            controls,
            variable=self.duration_var,
            from_=0.5,
            to=10.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            length=150,
            showvalue=True,
            command=self._on_duration_changed,
        )
        self.duration_scale.grid(row=0, column=8, padx=(4, 18))

        ttk.Button(controls, text="Cargar WAV", command=self.load_audio).grid(row=0, column=9, padx=3)
        ttk.Button(controls, text="Actualizar", command=self.update_simulation).grid(row=0, column=10, padx=3)
        ttk.Button(controls, text="Detener", command=self.stop_playback).grid(row=0, column=11, padx=3)

        ttk.Label(controls, textvariable=self.info_var).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(8, 0)
        )
        ttk.Label(controls, textvariable=self.ts_var).grid(row=1, column=4, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(controls, textvariable=self.step_var).grid(row=1, column=6, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=1, column=9, columnspan=3, sticky="e", pady=(8, 0)
        )

        zoom_frame = ttk.LabelFrame(controls, text="Zoom temporal", padding=5)
        zoom_frame.grid(row=2, column=0, columnspan=12, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(zoom_frame, text="Activar zoom", variable=self.zoom_enabled_var,
                        command=self._toggle_zoom).grid(row=0, column=0, padx=(0, 10))
        ttk.Label(zoom_frame, text="Inicio (s):").grid(row=0, column=1, sticky="w")
        self.zoom_start_entry = ttk.Entry(zoom_frame, textvariable=self.zoom_start_var, width=9)
        self.zoom_start_entry.grid(row=0, column=2, padx=(4, 12))
        ttk.Label(zoom_frame, text="Final (s):").grid(row=0, column=3, sticky="w")
        self.zoom_end_entry = ttk.Entry(zoom_frame, textvariable=self.zoom_end_var, width=9)
        self.zoom_end_entry.grid(row=0, column=4, padx=(4, 12))
        self.zoom_apply_button = ttk.Button(zoom_frame, text="Aplicar zoom", command=self.apply_zoom)
        self.zoom_apply_button.grid(row=0, column=5, padx=3)
        self.zoom_reset_button = ttk.Button(zoom_frame, text="Restablecer", command=self.reset_zoom)
        self.zoom_reset_button.grid(row=0, column=6, padx=3)
        ttk.Label(zoom_frame, text="Mismo intervalo en las 4 gráficas temporales.").grid(row=0, column=7, padx=(12, 0), sticky="w")
        self._set_zoom_controls_enabled(False)

        metrics = ttk.LabelFrame(self.root, text="Métricas", padding=8)
        metrics.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 6))

        ttk.Label(metrics, textvariable=self.reference_info_var).grid(row=0, column=0, padx=4)
        ttk.Label(metrics, text="MSE cuantificación").grid(row=0, column=1, padx=4)
        ttk.Label(metrics, textvariable=self.reference_mse_var, width=15).grid(row=0, column=2, padx=4)
        ttk.Label(metrics, text="SQNR cuantificación (dB)").grid(row=0, column=3, padx=4)
        ttk.Label(metrics, textvariable=self.reference_sqnr_var, width=15).grid(row=0, column=4, padx=4)
        ttk.Label(metrics, text="Error espectral").grid(row=0, column=5, padx=4)
        ttk.Label(metrics, textvariable=self.reference_spectral_var, width=15).grid(row=0, column=6, padx=4)

        ttk.Label(metrics, text="Seleccionada / reconstruida").grid(row=1, column=0, padx=4)
        ttk.Label(metrics, text="MSE cuantificación").grid(row=1, column=1, padx=4)
        ttk.Label(metrics, textvariable=self.mse_var, width=15).grid(row=1, column=2, padx=4)
        ttk.Label(metrics, text="SQNR cuantificación (dB)").grid(row=1, column=3, padx=4)
        ttk.Label(metrics, textvariable=self.sqnr_var, width=15).grid(row=1, column=4, padx=4)
        ttk.Label(metrics, text="Error espectral").grid(row=1, column=5, padx=4)
        ttk.Label(metrics, textvariable=self.spectral_var, width=15).grid(row=1, column=6, padx=4)

        ttk.Button(metrics, text="Audio original", command=self.play_original).grid(row=0, column=7, padx=4)
        ttk.Button(metrics, text="Audio reconstruido", command=self.play_reconstructed).grid(row=1, column=7, padx=4)
        ttk.Button(metrics, text="Exportar tablas", command=self.export_tables).grid(
            row=0, column=8, rowspan=2, padx=4
        )

    def _build_plots(self) -> None:
        self.figure, axes = plt.subplots(3, 2, figsize=(12, 7), constrained_layout=True)
        self.axes = axes.ravel()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.figure.suptitle("Simulación de un sistema de digitalización", fontsize=14)

    def _on_signal_changed(self, _event=None) -> None:
        is_audio = self.signal_var.get().startswith("D")
        if is_audio:
            # The time/duration slider must remain usable for audio.  It controls
            # the portion of the loaded WAV used by the current simulation.
            self.duration_scale.configure(state=tk.NORMAL)
            if self.audio is not None:
                self.fs_var.set(float(self.audio.sample_rate))
                self.fs_entry_var.set(str(int(self.audio.sample_rate)))
                maximum_duration = max(0.5, float(self.audio.duration))
                self.duration_scale.configure(from_=0.5, to=maximum_duration)
                current_duration = float(self.duration_var.get())
                self.duration_var.set(min(max(current_duration, 0.5), maximum_duration))
            else:
                self.duration_scale.configure(from_=0.5, to=20.0)
                self.duration_var.set(2.0)
        else:
            self.duration_scale.configure(state=tk.NORMAL)
            self.fs_var.set(100.0)
            self.fs_entry_var.set("100")
            self.duration_scale.configure(from_=0.5, to=10.0)
            self.duration_var.set(2.0)
        self._schedule_update()

    def _set_zoom_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in (self.zoom_start_entry, self.zoom_end_entry, self.zoom_apply_button, self.zoom_reset_button):
            widget.configure(state=state)

    def _toggle_zoom(self) -> None:
        enabled = bool(self.zoom_enabled_var.get())
        if enabled:
            duration = float(self.duration_var.get())
            self.zoom_start_var.set("0")
            self.zoom_end_var.set(f"{duration:g}")
        self._set_zoom_controls_enabled(enabled)
        if self.result is not None:
            self._show_result(self.result)

    def _on_duration_changed(self, _value=None) -> None:
        self.zoom_enabled_var.set(False)
        self._set_zoom_controls_enabled(False)
        self.zoom_start_var.set("0")
        self.zoom_end_var.set(f"{float(self.duration_var.get()):g}")
        self._schedule_update()

    def _validate_zoom(self):
        start = float(self.zoom_start_var.get().replace(",", "."))
        end = float(self.zoom_end_var.get().replace(",", "."))
        duration = float(self.duration_var.get())
        if start < 0 or end > duration or start >= end:
            raise ValueError(f"El zoom debe cumplir 0 ≤ inicio < final ≤ {duration:g} s")
        return start, end

    def apply_zoom(self) -> None:
        if not self.zoom_enabled_var.get():
            return
        try:
            start, end = self._validate_zoom()
            self.zoom_start_var.set(f"{start:g}")
            self.zoom_end_var.set(f"{end:g}")
            if self.result is not None:
                self._show_result(self.result)
        except ValueError as error:
            messagebox.showerror("Zoom temporal", str(error))

    def reset_zoom(self) -> None:
        duration = float(self.duration_var.get())
        self.zoom_start_var.set("0")
        self.zoom_end_var.set(f"{duration:g}")
        self.zoom_enabled_var.set(False)
        self._set_zoom_controls_enabled(False)
        if self.result is not None:
            self._show_result(self.result)

    def apply_frequency(self) -> None:
        try:
            value = float(self.fs_entry_var.get().replace(",", "."))
            if not np.isfinite(value) or value <= 0:
                raise ValueError("La frecuencia de muestreo debe ser positiva")
            if self.signal_var.get().startswith("D") and not float(value).is_integer():
                raise ValueError("Para audio, fs debe ser un número entero de Hz")
            self.fs_var.set(value)
            self.fs_entry_var.set(f"{value:g}")
            self._schedule_update()
        except ValueError as error:
            messagebox.showerror("Frecuencia de muestreo", str(error))

    def _schedule_update(self) -> None:
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(250, self.update_simulation)

    def _parse_controls(self) -> tuple[float, int, float]:
        fs = float(self.fs_entry_var.get().replace(",", "."))
        self.fs_var.set(fs)
        levels = int(self.levels_var.get())
        duration = float(self.duration_var.get())
        if fs <= 0 or duration <= 0:
            raise ValueError("fs y la duración deben ser positivos")
        if levels not in VALID_LEVELS:
            raise ValueError("los niveles deben ser una potencia de 2 permitida")
        return fs, levels, duration

    def load_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleccionar archivo WAV",
            filetypes=(("Audio WAV", "*.wav"), ("Todos los archivos", "*.*")),
        )
        if not path:
            return
        try:
            self.audio = load_wav(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Error de audio", str(error))
            return

        self.signal_var.set("D - Audio")
        self.info_var.set(
            f"Audio: {self.audio.sample_rate} Hz | {self.audio.duration:.2f} s | "
            f"{self.audio.channels} canal(es) | PCM {self.audio.sample_width * 8} bits"
        )
        self._on_signal_changed()
        self.duration_scale.configure(state=tk.NORMAL)
        self.status_var.set("Audio cargado correctamente. La barra de tiempo está habilitada.")

    def update_simulation(self) -> None:
        self.update_job = None
        try:
            fs, levels, duration = self._parse_controls()
            if self.signal_var.get().startswith("D"):
                if self.audio is None:
                    raise ValueError("cargue primero un archivo WAV de mínimo 20 segundos")

                # Use exactly the duration selected by the time slider.
                # The original loaded WAV remains intact; only the simulation
                # window changes.
                frame_count = max(2, int(round(duration * self.audio.sample_rate)))
                frame_count = min(frame_count, self.audio.samples.size)
                audio_window = AudioSignal(
                    samples=self.audio.samples[:frame_count].copy(),
                    sample_rate=self.audio.sample_rate,
                    channels=self.audio.channels,
                    sample_width=self.audio.sample_width,
                    path=self.audio.path,
                )
                result = simulate_audio(audio_window, fs, levels)
            else:
                result = simulate_synthetic(
                    self.signal_var.get(), duration, 2000.0, fs, levels
                )

            self.result = result
            self._show_result(result)
            self.ts_var.set(f"Ts = {1.0 / fs:.9f} s")
            self.step_var.set(f"Δ = {result.quantization_step:.8g}")
            self.status_var.set(f"fs = {fs:g} Hz | N = {levels}")
        except (OSError, ValueError, RuntimeError) as error:
            self.status_var.set(f"Error: {error}")

    @staticmethod
    def _display_indices(size: int, maximum: int = 10000):
        if size <= maximum:
            return slice(None)
        step = int(np.ceil(size / maximum))
        return slice(None, None, step)

    def _show_result(self, result: SimulationResult) -> None:
        for axis in self.axes:
            axis.clear()

        time_idx = self._display_indices(result.time.size, 10000)
        time = result.time[time_idx]
        original = result.original[time_idx]
        reconstructed = result.reconstructed[time_idx]

        self.axes[0].plot(time, original, color="tab:blue", linewidth=1.4)
        self.axes[0].set_title("Señal original en el tiempo")
        self.axes[0].set_xlabel("Tiempo (s)")
        self.axes[0].set_ylabel("Amplitud")

        spec_idx = self._display_indices(result.original_spectrum_frequency.size, 12000)
        self.axes[1].plot(result.original_spectrum_frequency[spec_idx], result.original_spectrum[spec_idx],
                          color="tab:purple", linewidth=1.2)
        self.axes[1].set_title("Espectro de la señal original")
        self.axes[1].set_xlabel("Frecuencia (Hz)")
        self.axes[1].set_ylabel("Magnitud")

        sampled_idx = self._display_indices(result.sampled.time.size, 5000)
        markerline, stemlines, baseline = self.axes[2].stem(
            result.sampled.time[sampled_idx], result.sampled.values[sampled_idx],
            linefmt="tab:orange", markerfmt="o", basefmt=" "
        )
        markerline.set_markersize(4.0)
        stemlines.set_linewidth(1.0)
        self.axes[2].set_title("Señal muestreada")
        self.axes[2].set_xlabel("Tiempo (s)")
        self.axes[2].set_ylabel("Amplitud")

        sampled_spec_idx = self._display_indices(result.sampled_spectrum_frequency.size, 12000)
        self.axes[3].plot(result.sampled_spectrum_frequency[sampled_spec_idx], result.sampled_spectrum[sampled_spec_idx],
                          color="tab:green", linewidth=1.2)
        self.axes[3].set_title("Espectro de la señal muestreada")
        self.axes[3].set_xlabel("Frecuencia (Hz)")
        self.axes[3].set_ylabel("Magnitud")

        self.axes[4].step(result.sampled.time[sampled_idx], result.quantized.values[sampled_idx],
                          where="mid", color="tab:red", linewidth=1.2)
        self.axes[4].set_title("Señal cuantificada")
        self.axes[4].set_xlabel("Tiempo (s)")
        self.axes[4].set_ylabel("Amplitud")

        self.axes[5].plot(time, reconstructed, color="tab:blue", linewidth=1.4)
        self.axes[5].set_title("Señal reconstruida")
        self.axes[5].set_xlabel("Tiempo (s)")
        self.axes[5].set_ylabel("Amplitud")

        for axis in self.axes:
            axis.grid(True, alpha=0.25)

        if self.zoom_enabled_var.get():
            try:
                start_zoom, end_zoom = self._validate_zoom()
                for axis in (self.axes[0], self.axes[2], self.axes[4], self.axes[5]):
                    axis.set_xlim(start_zoom, end_zoom)
            except ValueError:
                self.zoom_enabled_var.set(False)
                self._set_zoom_controls_enabled(False)

        self.mse_var.set(f"{result.mse:.8g}")
        self.sqnr_var.set("∞" if np.isinf(result.sqnr) else f"{result.sqnr:.8g}")
        self.spectral_var.set(f"{result.spectral_error:.8g}")
        if result.is_audio and result.reference_mse is not None:
            self.reference_info_var.set(f"Referencia PCM ({result.reference_levels} niveles)")
            self.reference_mse_var.set(f"{result.reference_mse:.8g}")
            self.reference_sqnr_var.set("∞" if np.isinf(result.reference_sqnr) else f"{result.reference_sqnr:.8g}")
            self.reference_spectral_var.set(f"{result.reference_spectral_error:.8g}")
        else:
            self.reference_info_var.set("Referencia WAV")
            self.reference_mse_var.set("-")
            self.reference_sqnr_var.set("-")
            self.reference_spectral_var.set("-")
        self.canvas.draw_idle()

    def export_tables(self) -> None:
        directory = filedialog.askdirectory(title="Carpeta para guardar resultados")
        if not directory:
            return
        try:
            run_and_export_quantization(directory)
            if self.audio is None:
                raise ValueError("cargue el WAV para exportar la tabla de audio")
            run_and_export_sampling(self.audio, directory, quantization_levels=16)
            self.status_var.set("Tabla 1, Tabla 2 y gráficas exportadas correctamente")
            messagebox.showinfo(
                "Exportación completada",
                "Se generaron los CSV y PNG de las dos tablas en la carpeta seleccionada.",
            )
        except (OSError, ValueError, RuntimeError) as error:
            messagebox.showerror("Error al exportar", str(error))

    def play_original(self) -> None:
        if self.audio is None:
            messagebox.showinfo("Audio", "Cargue primero un archivo WAV")
            return
        try:
            play_audio(self.audio.samples, self.audio.sample_rate)
        except (RuntimeError, ValueError) as error:
            messagebox.showerror("Reproducción", str(error))

    def play_reconstructed(self) -> None:
        if self.result is None or not self.result.is_audio:
            messagebox.showinfo("Audio", "Actualice primero la simulación de audio")
            return
        try:
            samples = self.result.playback_reconstructed
            rate = self.result.playback_sample_rate
            if samples is None or rate is None:
                raise RuntimeError("no existe una señal reconstruida para reproducción")
            play_audio(samples, rate)
        except (RuntimeError, ValueError) as error:
            messagebox.showerror("Reproducción", str(error))

    def stop_playback(self) -> None:
        try:
            stop_audio()
        except RuntimeError as error:
            messagebox.showerror("Reproducción", str(error))


def main() -> None:
    root = tk.Tk()
    DigitizationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
