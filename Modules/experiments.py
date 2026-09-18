"""Experimentos reproducibles requeridos por la especificación del proyecto."""

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .audio_io import AudioSignal
from .simulation import simulate_audio, simulate_synthetic


QUANTIZATION_LEVELS = (2, 4, 8, 16, 32, 64, 128, 256)
SAMPLING_FACTORS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


@dataclass(frozen=True)
class QuantizationRow:
    levels: int
    mse: float
    sqnr_db: float


@dataclass(frozen=True)
class SamplingRow:
    factor: float
    sampling_frequency: float
    spectral_error: float


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)


def quantization_experiment(
    duration: float = 2.0,
    representation_fs: float = 2000.0,
    nyquist_fs: float = 10.0,
) -> list[QuantizationRow]:
    """Tabla 1: señal C a su frecuencia de muestreo de Nyquist."""

    rows = []
    for levels in QUANTIZATION_LEVELS:
        result = simulate_synthetic(
            "C",
            duration=duration,
            representation_fs=representation_fs,
            sampling_fs=nyquist_fs,
            levels=levels,
        )
        rows.append(QuantizationRow(levels, result.mse, result.sqnr))
    return rows


def sampling_frequency_experiment(
    audio: AudioSignal,
    quantization_levels: int = 16,
) -> list[SamplingRow]:
    """Tabla 2: señal D a las ocho frecuencias relativas de muestreo requeridas."""

    rows = []
    for factor in SAMPLING_FACTORS:
        frequency = int(round(audio.sample_rate * factor))
        result = simulate_audio(audio, frequency, quantization_levels)
        rows.append(SamplingRow(factor, frequency, result.spectral_error))
    return rows


def export_quantization_results(
    rows: Sequence[QuantizationRow], output_directory: str = "results"
) -> tuple[Path, Path]:
    directory = Path(output_directory)
    csv_path = directory / "tabla_1_cuantificacion.csv"
    image_path = directory / "tabla_1_cuantificacion.png"
    _write_csv(
        csv_path,
        ("niveles_cuantificacion", "mse", "sqnr_db"),
        ((row.levels, row.mse, row.sqnr_db) for row in rows),
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    levels = [row.levels for row in rows]
    axes[0].semilogx(levels, [row.mse for row in rows], marker="o")
    axes[0].set_title("MSE vs niveles de cuantificación")
    axes[0].set_xlabel("Niveles de cuantificación")
    axes[0].set_ylabel("MSE")
    axes[1].semilogx(levels, [row.sqnr_db for row in rows], marker="o")
    axes[1].set_title("SQNR vs niveles de cuantificación")
    axes[1].set_xlabel("Niveles de cuantificación")
    axes[1].set_ylabel("SQNR (dB)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.savefig(image_path, dpi=150)
    plt.close(figure)
    return csv_path, image_path


def export_sampling_results(
    rows: Sequence[SamplingRow], output_directory: str = "results"
) -> tuple[Path, Path]:
    directory = Path(output_directory)
    csv_path = directory / "tabla_2_frecuencia_muestreo.csv"
    image_path = directory / "tabla_2_frecuencia_muestreo.png"
    _write_csv(
        csv_path,
        ("factor_fs", "frecuencia_muestreo_hz", "error_espectral"),
        ((row.factor, row.sampling_frequency, row.spectral_error) for row in rows),
    )

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    factors = [row.factor for row in rows]
    axis.plot(factors, [row.spectral_error for row in rows], marker="o")
    axis.set_title("Error espectral vs frecuencia de muestreo")
    axis.set_xlabel("Frecuencia de muestreo / frecuencia original")
    axis.set_ylabel("Error espectral relativo")
    axis.set_xticks(factors)
    axis.grid(True, alpha=0.25)
    figure.savefig(image_path, dpi=150)
    plt.close(figure)
    return csv_path, image_path


def run_and_export_quantization(
    output_directory: str = "results",
    duration: float = 2.0,
    representation_fs: float = 2000.0,
    nyquist_fs: float = 10.0,
) -> list[QuantizationRow]:
    rows = quantization_experiment(duration, representation_fs, nyquist_fs)
    export_quantization_results(rows, output_directory)
    return rows


def run_and_export_sampling(
    audio: AudioSignal,
    output_directory: str = "results",
    quantization_levels: int = 16,
) -> list[SamplingRow]:
    rows = sampling_frequency_experiment(audio, quantization_levels)
    export_sampling_results(rows, output_directory)
    return rows


__all__ = [
    "QUANTIZATION_LEVELS",
    "SAMPLING_FACTORS",
    "QuantizationRow",
    "SamplingRow",
    "export_quantization_results",
    "export_sampling_results",
    "quantization_experiment",
    "run_and_export_quantization",
    "run_and_export_sampling",
    "sampling_frequency_experiment",
]
