"""Pipelines de simulación a nivel de aplicación."""

from dataclasses import dataclass
import numpy as np

from .audio_io import AudioSignal
from .dsp_core import (
    SampledSignal,
    calculate_metrics,
    one_sided_spectrum,
    pulse_train,
    quantize_uniform,
    reconstruct_zero_order_hold,
    resample_linear,
    sample_ideal_pulses,
    source_quantization_metrics,
    validate_levels,
    validate_sampling_frequency,
)
from .signals import create_time_axis, generate_signal


@dataclass(frozen=True)
class SimulationResult:
    time: np.ndarray
    original: np.ndarray
    original_fs: float
    sampled: object
    sampled_grid: np.ndarray
    quantized: object
    reconstructed: np.ndarray
    original_spectrum_frequency: np.ndarray
    original_spectrum: np.ndarray
    sampled_spectrum_frequency: np.ndarray
    sampled_spectrum: np.ndarray
    mse: float
    sqnr: float
    spectral_error: float
    is_audio: bool
    source_sample_rate: float
    playback_reconstructed: np.ndarray | None = None
    playback_sample_rate: float | None = None
    reference_mse: float | None = None
    reference_sqnr: float | None = None
    reference_spectral_error: float | None = None
    reference_levels: int | None = None
    quantization_step: float = 0.0


def _validate_signal_inputs(time: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_array = np.asarray(time, dtype=float)
    signal_array = np.asarray(signal, dtype=float)
    if time_array.ndim != 1 or signal_array.ndim != 1:
        raise ValueError("time and signal must be one-dimensional")
    if time_array.size != signal_array.size or time_array.size < 2:
        raise ValueError("time and signal must have the same length and at least two samples")
    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must be strictly increasing")
    if not np.all(np.isfinite(signal_array)):
        raise ValueError("signal contains non-finite values")
    return time_array, signal_array


def run_simulation(
    time: np.ndarray,
    signal: np.ndarray,
    original_fs: float,
    sampling_fs: float,
    levels: int,
    is_audio: bool = False,
) -> SimulationResult:
    """Ejecuta muestreo, cuantización, reconstrucción ZOH, FFT y métricas."""

    time_array, signal_array = _validate_signal_inputs(time, signal)
    source_fs = validate_sampling_frequency(original_fs)
    target_fs = validate_sampling_frequency(sampling_fs)
    level_count = validate_levels(levels)

    sampled = sample_ideal_pulses(time_array, signal_array, target_fs)
    quantized = quantize_uniform(sampled.values, level_count)
    reconstructed = reconstruct_zero_order_hold(sampled.time, quantized.values, time_array)
    sampled_grid = pulse_train(time_array, sampled)

    original_frequency, original_spectrum = one_sided_spectrum(signal_array, source_fs)
    sampled_frequency, sampled_spectrum = one_sided_spectrum(sampled_grid, source_fs)
    mse, sqnr, _ = calculate_metrics(sampled.values, quantized.values, target_fs)
    _, _, spectral_error = calculate_metrics(signal_array, reconstructed, source_fs)

    return SimulationResult(
        time=time_array,
        original=signal_array,
        original_fs=source_fs,
        sampled=sampled,
        sampled_grid=sampled_grid,
        quantized=quantized,
        reconstructed=reconstructed,
        original_spectrum_frequency=original_frequency,
        original_spectrum=original_spectrum,
        sampled_spectrum_frequency=sampled_frequency,
        sampled_spectrum=sampled_spectrum,
        mse=mse,
        sqnr=sqnr,
        spectral_error=spectral_error,
        is_audio=is_audio,
        source_sample_rate=source_fs,
        quantization_step=quantized.step,
    )


def simulate_synthetic(
    name: str,
    duration: float = 2.0,
    representation_fs: float = 2000.0,
    sampling_fs: float = 100.0,
    levels: int = 16,
) -> SimulationResult:
    time = create_time_axis(duration, representation_fs)
    signal = generate_signal(name, time)
    return run_simulation(time, signal, representation_fs, sampling_fs, levels)


def simulate_audio(audio: AudioSignal, sampling_fs: float, levels: int = 16) -> SimulationResult:
    """Remuestrea explícitamente el audio, lo requantiza y lo compara a la tasa original.

    Cadena de procesamiento:
        WAV original -> remuestreo -> cuantización uniforme -> reconstrucción tipo
        ZOH a la tasa objetivo -> remuestreo de regreso a la tasa original.

    La ``reconstructed`` devuelta está en la malla original del audio para que MSE,
    SQNR y error espectral se calculen con arreglos de la misma frecuencia de
    muestreo. ``playback_reconstructed`` permanece a la tasa objetivo seleccionada.
    """

    target_fs = validate_sampling_frequency(sampling_fs)
    if not np.isclose(target_fs, round(target_fs)):
        raise ValueError("audio sampling frequency must be an integer number of Hz")
    target_fs_int = int(round(target_fs))

    original = np.asarray(audio.samples, dtype=float)
    source_fs = float(audio.sample_rate)

    # Explicit resampling required by the project.
    resampled = resample_linear(original, source_fs, target_fs_int)

    # Requantification at the user-selected number of levels.
    quantized = quantize_uniform(resampled, levels, -1.0, 1.0)

    # Quantized samples are the digital samples at target_fs.  ZOH reconstructs
    # the continuous-time-like waveform on a dense target-rate grid.
    target_time = np.arange(quantized.values.size, dtype=float) / target_fs_int
    source_time = np.arange(original.size, dtype=float) / source_fs
    reconstructed_target = quantized.values.copy()

    # Return to the original grid for objective comparison and spectral error.
    reconstructed_source = resample_linear(reconstructed_target, target_fs_int, source_fs)
    if reconstructed_source.size < original.size:
        reconstructed_source = np.pad(
            reconstructed_source,
            (0, original.size - reconstructed_source.size),
            mode="edge",
        )
    elif reconstructed_source.size > original.size:
        reconstructed_source = reconstructed_source[: original.size]

    mse, sqnr, _ = calculate_metrics(resampled, quantized.values, target_fs_int)
    _, _, spectral_error = calculate_metrics(original, reconstructed_source, source_fs)
    reference_mse, reference_sqnr, reference_spectral, reference_levels = source_quantization_metrics(
        original, audio.sample_width
    )

    # Sampled representation for the six GUI plots: target-rate samples.
    sampled_time = target_time
    sampled_values = resampled
    sampled = SampledSignal(
        time=sampled_time,
        values=sampled_values,
        frequency=float(target_fs_int),
    )

    # Dense source-rate grid for the visualization of sampled impulses.
    sampled_grid = np.zeros_like(original)
    source_indices = np.rint(sampled_time * source_fs).astype(int)
    valid = (source_indices >= 0) & (source_indices < original.size)
    sampled_grid[source_indices[valid]] = sampled_values[valid]

    original_frequency, original_spectrum = one_sided_spectrum(original, source_fs)
    sampled_frequency, sampled_spectrum = one_sided_spectrum(sampled_grid, source_fs)

    # QuantizedSignal-like object for GUI compatibility.
    quantized_object = quantized

    return SimulationResult(
        time=source_time,
        original=original,
        original_fs=source_fs,
        sampled=sampled,
        sampled_grid=sampled_grid,
        quantized=quantized_object,
        reconstructed=reconstructed_source,
        original_spectrum_frequency=original_frequency,
        original_spectrum=original_spectrum,
        sampled_spectrum_frequency=sampled_frequency,
        sampled_spectrum=sampled_spectrum,
        mse=mse,
        sqnr=sqnr,
        spectral_error=spectral_error,
        is_audio=True,
        source_sample_rate=source_fs,
        playback_reconstructed=reconstructed_target,
        playback_sample_rate=float(target_fs_int),
        reference_mse=reference_mse,
        reference_sqnr=reference_sqnr,
        reference_spectral_error=reference_spectral,
        reference_levels=reference_levels,
        quantization_step=quantized.step,
    )


__all__ = ["SimulationResult", "run_simulation", "simulate_audio", "simulate_synthetic"]
