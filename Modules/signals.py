"""Señales sintéticas utilizadas por el simulador de digitalización.

Las constantes se mantienen fijas porque la especificación del proyecto indica
que A, f0 y a son constantes de diseño y no parámetros definidos por el usuario.
"""

from typing import Callable, Dict

import numpy as np


SIGNAL_A = "A"
SIGNAL_B = "B"
SIGNAL_C = "C"

DEFAULT_AMPLITUDE = 1.0
DEFAULT_TONE_FREQUENCY = 10.0
DEFAULT_SINC_PARAMETER = 5.0


def _as_time_array(time: np.ndarray) -> np.ndarray:
    time_array = np.asarray(time, dtype=float)
    if time_array.ndim != 1 or time_array.size == 0:
        raise ValueError("time must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(time_array)):
        raise ValueError("time contains non-finite values")
    return time_array


def normalized_sinc(argument: np.ndarray) -> np.ndarray:
    """Evalúa sinc(u) = sin(pi*u)/(pi*u), incluyendo sinc(0) = 1."""

    values = np.asarray(argument, dtype=float)
    result = np.ones_like(values, dtype=float)
    nonzero = values != 0.0
    scaled_values = np.pi * values[nonzero]
    result[nonzero] = np.sin(scaled_values) / scaled_values
    return result


def signal_a(
    time: np.ndarray,
    amplitude: float = DEFAULT_AMPLITUDE,
    frequency: float = DEFAULT_TONE_FREQUENCY,
) -> np.ndarray:
    """Genera y(t) = A*cos(2*pi*f0*t)."""

    time_array = _as_time_array(time)
    amplitude_value = float(amplitude)
    frequency_value = float(frequency)
    if not np.isfinite(amplitude_value) or not np.isfinite(frequency_value):
        raise ValueError("amplitude and frequency must be finite")
    return amplitude_value * np.cos(2.0 * np.pi * frequency_value * time_array)


def signal_b(
    time: np.ndarray, parameter: float = DEFAULT_SINC_PARAMETER
) -> np.ndarray:
    """Genera x(t) = sinc(2*a*t) usando la definición normalizada de sinc."""

    time_array = _as_time_array(time)
    parameter_value = float(parameter)
    if not np.isfinite(parameter_value):
        raise ValueError("parameter must be finite")
    return normalized_sinc(2.0 * parameter_value * time_array)


def signal_c(
    time: np.ndarray, parameter: float = DEFAULT_SINC_PARAMETER
) -> np.ndarray:
    """Genera z(t) = sinc^2(a*t) + x(t)."""

    time_array = _as_time_array(time)
    parameter_value = float(parameter)
    if not np.isfinite(parameter_value):
        raise ValueError("parameter must be finite")
    x_values = signal_b(time_array, parameter_value)
    sinc_values = normalized_sinc(parameter_value * time_array)
    return sinc_values * sinc_values + x_values


def available_signals() -> tuple[str, ...]:
    """Devuelve los identificadores de las señales sintéticas en el orden de visualización de la GUI."""

    return SIGNAL_A, SIGNAL_B, SIGNAL_C


def generate_signal(name: str, time: np.ndarray) -> np.ndarray:
    """Genera una señal sintética seleccionada por su identificador."""

    generators: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        SIGNAL_A: signal_a,
        SIGNAL_B: signal_b,
        SIGNAL_C: signal_c,
    }
    try:
        generator = generators[name.upper()]
    except (AttributeError, KeyError) as error:
        allowed = ", ".join(available_signals())
        raise ValueError(f"unknown synthetic signal; choose: {allowed}") from error
    return generator(time)


def create_time_axis(duration: float = 2.0, representation_fs: float = 2000.0) -> np.ndarray:
    """Crea la malla temporal densa usada para representar una señal tipo analógica."""

    duration_value = float(duration)
    frequency_value = float(representation_fs)
    if not np.isfinite(duration_value) or duration_value <= 0.0:
        raise ValueError("duration must be a positive finite number")
    if not np.isfinite(frequency_value) or frequency_value <= 0.0:
        raise ValueError("representation_fs must be a positive finite number")
    sample_count = int(np.floor(duration_value * frequency_value)) + 1
    return np.arange(sample_count, dtype=float) / frequency_value


__all__ = [
    "DEFAULT_AMPLITUDE",
    "DEFAULT_SINC_PARAMETER",
    "DEFAULT_TONE_FREQUENCY",
    "SIGNAL_A",
    "SIGNAL_B",
    "SIGNAL_C",
    "available_signals",
    "create_time_axis",
    "generate_signal",
    "normalized_sinc",
    "signal_a",
    "signal_b",
    "signal_c",
]
