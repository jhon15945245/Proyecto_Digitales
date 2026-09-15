"""Core numerical routines for the signal digitization simulator.

The project is implemented with NumPy and the Python standard library.  No
high-level DSP resampling/quantization functions are used.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


VALID_LEVELS = (2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class SampledSignal:
    time: np.ndarray
    values: np.ndarray
    frequency: float


@dataclass(frozen=True)
class QuantizedSignal:
    values: np.ndarray
    levels: int
    minimum: float
    maximum: float
    step: float


@dataclass(frozen=True)
class AudioProcessingResult:
    """Audio after explicit resampling and uniform quantization."""

    resampled: np.ndarray
    quantized: np.ndarray
    reconstructed: np.ndarray
    target_fs: float


def _as_1d_float(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_sampling_frequency(fs: float) -> float:
    frequency = float(fs)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("fs must be a positive finite number")
    return frequency


def validate_levels(levels: int) -> int:
    integer_levels = int(levels)
    if integer_levels != levels or integer_levels not in VALID_LEVELS:
        allowed = ", ".join(str(value) for value in VALID_LEVELS)
        raise ValueError(f"levels must be one of: {allowed}")
    return integer_levels


def _validate_time_axis(time: np.ndarray) -> np.ndarray:
    time_array = _as_1d_float(time, "time")
    if time_array.size < 2 or np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must contain at least two strictly increasing values")
    return time_array


def _linear_interpolate(
    source_time: np.ndarray, source_values: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    """Linear interpolation implemented explicitly with index arithmetic."""

    source_time = _validate_time_axis(source_time)
    source_values = _as_1d_float(source_values, "source_values")
    target_time = _as_1d_float(target_time, "target_time")
    if source_time.size != source_values.size:
        raise ValueError("source_time and source_values must have the same length")

    right_indices = np.searchsorted(source_time, target_time, side="left")
    right_indices = np.clip(right_indices, 1, source_time.size - 1)
    left_indices = right_indices - 1
    left_time = source_time[left_indices]
    right_time = source_time[right_indices]
    fraction = (target_time - left_time) / (right_time - left_time)
    return source_values[left_indices] + fraction * (
        source_values[right_indices] - source_values[left_indices]
    )


def sample_ideal_pulses(time: np.ndarray, signal: np.ndarray, fs: float) -> SampledSignal:
    """Obtain samples x(nTs) at the selected sampling frequency."""

    time_array = _validate_time_axis(time)
    signal_array = _as_1d_float(signal, "signal")
    if signal_array.size != time_array.size:
        raise ValueError("time and signal must have the same length")

    frequency = validate_sampling_frequency(fs)
    duration = time_array[-1] - time_array[0]
    sample_count = int(np.floor(duration * frequency + 1e-12)) + 1
    sample_times = time_array[0] + np.arange(sample_count, dtype=float) / frequency
    sample_times = sample_times[sample_times <= time_array[-1] + 1e-12]
    sample_values = _linear_interpolate(time_array, signal_array, sample_times)
    return SampledSignal(sample_times, sample_values, frequency)


def pulse_train(time: np.ndarray, sampled: SampledSignal) -> np.ndarray:
    """Represent the ideal-sampling impulses on a dense display grid.

    Each sample is shown at its nearest representation-grid position.  The
    result is intended for visualization of the sampled signal, not as a
    continuous-time Dirac-delta distribution.
    """

    time_array = _validate_time_axis(time)
    output = np.zeros(time_array.size, dtype=float)
    indices = np.searchsorted(time_array, sampled.time)
    right_indices = np.clip(indices, 0, time_array.size - 1)
    left_indices = np.clip(right_indices - 1, 0, time_array.size - 1)
    choose_left = np.abs(time_array[left_indices] - sampled.time) < np.abs(
        time_array[right_indices] - sampled.time
    )
    closest = np.where(choose_left, left_indices, right_indices)
    tolerance = 0.5 * np.min(np.diff(time_array)) + 1e-12
    valid = np.abs(time_array[closest] - sampled.time) <= tolerance
    output[closest[valid]] = sampled.values[valid]
    return output


def resample_linear(signal: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Manually resample a mono signal by linear interpolation.

    This deliberately does not call scipy.signal.resample or an equivalent
    high-level DSP routine.  No anti-alias filter is hidden inside the method,
    which is useful here because the project studies the effect of sampling
    frequency and spectral error.
    """

    signal_array = _as_1d_float(signal, "signal")
    source_frequency = validate_sampling_frequency(original_fs)
    target_frequency = validate_sampling_frequency(target_fs)
    if signal_array.size == 1:
        return signal_array.copy()

    duration = (signal_array.size - 1) / source_frequency
    target_count = max(1, int(np.floor(duration * target_frequency + 1e-12)) + 1)
    source_time = np.arange(signal_array.size, dtype=float) / source_frequency
    target_time = np.arange(target_count, dtype=float) / target_frequency
    target_time = np.minimum(target_time, source_time[-1])
    return _linear_interpolate(source_time, signal_array, target_time)


def quantize_uniform(
    signal: np.ndarray, levels: int, minimum: float = None, maximum: float = None
) -> QuantizedSignal:
    """Apply a uniform mid-rise quantizer."""

    signal_array = _as_1d_float(signal, "signal")
    level_count = validate_levels(levels)
    lower = float(np.min(signal_array) if minimum is None else minimum)
    upper = float(np.max(signal_array) if maximum is None else maximum)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("maximum must be greater than minimum")

    step = (upper - lower) / level_count
    indices = np.floor((signal_array - lower) / step).astype(np.int64)
    indices = np.clip(indices, 0, level_count - 1)
    quantized_values = lower + (indices + 0.5) * step
    return QuantizedSignal(quantized_values, level_count, lower, upper, step)


def reconstruct_zero_order_hold(
    sample_time: np.ndarray, sample_values: np.ndarray, time: np.ndarray
) -> np.ndarray:
    """Reconstruct by zero-order hold (ZOH)."""

    sample_times = _validate_time_axis(sample_time)
    values = _as_1d_float(sample_values, "sample_values")
    target_time = _as_1d_float(time, "time")
    if values.size != sample_times.size:
        raise ValueError("sample_time and sample_values must have the same length")

    indices = np.searchsorted(sample_times, target_time, side="right") - 1
    indices = np.clip(indices, 0, values.size - 1)
    return values[indices]


def one_sided_spectrum(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a one-sided amplitude spectrum with FFT."""

    values = _as_1d_float(signal, "signal")
    frequency = validate_sampling_frequency(fs)
    spectrum = np.fft.rfft(values)
    magnitude = np.abs(spectrum) / values.size
    if values.size > 1:
        if values.size % 2 == 0:
            magnitude[1:-1] *= 2.0
        else:
            magnitude[1:] *= 2.0
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / frequency)
    return frequencies, magnitude


def calculate_metrics(
    original: np.ndarray, reconstructed: np.ndarray, fs: float = None
) -> Tuple[float, float, float]:
    """Return MSE, SQNR (dB), and normalized RMS spectral error."""

    original_array = _as_1d_float(original, "original")
    reconstructed_array = _as_1d_float(reconstructed, "reconstructed")
    if original_array.size != reconstructed_array.size:
        raise ValueError("original and reconstructed must have the same length")

    error = original_array - reconstructed_array
    mse = float(np.mean(error * error))
    signal_power = float(np.mean(original_array * original_array))
    if mse == 0.0:
        sqnr = float("inf")
    elif signal_power == 0.0:
        sqnr = float("-inf")
    else:
        sqnr = float(10.0 * np.log10(signal_power / mse))

    original_fft = np.abs(np.fft.rfft(original_array))
    reconstructed_fft = np.abs(np.fft.rfft(reconstructed_array))
    spectral_difference = original_fft - reconstructed_fft
    denominator = float(np.sqrt(np.mean(original_fft * original_fft)))
    spectral_error = (
        float(np.sqrt(np.mean(spectral_difference * spectral_difference)) / denominator)
        if denominator > 0.0
        else 0.0
    )
    return mse, sqnr, spectral_error


def source_quantization_metrics(signal: np.ndarray, sample_width: int) -> Tuple[float, float, float, int]:
    """Estimate metrics associated with the PCM quantization resolution.

    The loaded WAV is already a digital signal, so an analog reference is not
    available.  We therefore report a transparent reference obtained by
    applying a uniform quantizer over [-1, 1] using the WAV's nominal PCM
    resolution.  This is a reproducible reference, not an estimate of the
    original microphone/analog waveform.
    """

    if sample_width not in (1, 2, 3, 4):
        raise ValueError("unsupported PCM sample width")
    levels = 2 ** (8 * sample_width)
    values = _as_1d_float(signal, "signal")
    lower, upper = -1.0, 1.0
    step = (upper - lower) / levels
    indices = np.floor((np.clip(values, lower, np.nextafter(upper, lower)) - lower) / step)
    indices = np.clip(indices.astype(np.int64), 0, levels - 1)
    reconstructed = lower + (indices + 0.5) * step
    mse, sqnr, spectral_error = calculate_metrics(values, reconstructed)
    return mse, sqnr, spectral_error, levels


__all__ = [
    "VALID_LEVELS",
    "SampledSignal",
    "QuantizedSignal",
    "AudioProcessingResult",
    "calculate_metrics",
    "one_sided_spectrum",
    "pulse_train",
    "quantize_uniform",
    "reconstruct_zero_order_hold",
    "resample_linear",
    "sample_ideal_pulses",
    "source_quantization_metrics",
    "validate_levels",
    "validate_sampling_frequency",
]
