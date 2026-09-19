"""Utilidades para cargar y reproducir WAV en el simulador de digitalización.

La decodificación de WAV usa solo la biblioteca estándar de Python. Las muestras
se exponen como arreglos mono en punto flotante en el rango [-1, 1]. El
remuestreo se delega a la implementación manual en NumPy de dsp_core.py.
"""

from dataclasses import dataclass
from pathlib import Path
import wave
import numpy as np

from .dsp_core import resample_linear, validate_sampling_frequency """Estas funciones son extraidas de dsp_core.py para remuestreo y validación de frecuencia de muestreo."""


MIN_AUDIO_DURATION_SECONDS = 20.0
SUPPORTED_SAMPLE_WIDTHS = (1, 2, 3, 4)


@dataclass(frozen=True)
class AudioSignal:
    """Audio mono decodificado y sus metadatos de origen."""

    samples: np.ndarray
    sample_rate: int
    channels: int
    sample_width: int
    path: str

    @property
    def duration(self) -> float:
        """Devuelve la duración en segundos."""

        return self.samples.size / self.sample_rate


def _decode_pcm(raw_data: bytes, sample_width: int) -> np.ndarray:
    """Decodifica bytes PCM little-endian en muestras flotantes en [-1, 1]."""

    if sample_width == 1:
        unsigned_values = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float64)
        return (unsigned_values - 128.0) / 128.0

    if sample_width == 2:
        values = np.frombuffer(raw_data, dtype="<i2").astype(np.float64)
        return values / 32768.0

    if sample_width == 3:
        bytes_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
        values = (
            bytes_array[:, 0].astype(np.int32)
            | (bytes_array[:, 1].astype(np.int32) << 8)
            | (bytes_array[:, 2].astype(np.int32) << 16)
        )
        negative = values & 0x800000
        values = values - (negative << 1)
        return values.astype(np.float64) / 8388608.0

    if sample_width == 4:
        values = np.frombuffer(raw_data, dtype="<i4").astype(np.float64)
        return values / 2147483648.0

    raise ValueError("only 8, 16, 24, and 32-bit PCM WAV files are supported")


def _to_mono(interleaved_samples: np.ndarray, channels: int) -> np.ndarray:
    """Convierte muestras entrelazadas a mono promediando los canales."""

    if channels == 1:
        return interleaved_samples.copy()
    if channels < 1:
        raise ValueError("the WAV file reports an invalid channel count")
    if interleaved_samples.size % channels != 0:
        raise ValueError("WAV data is incomplete for its channel count")
    frames = interleaved_samples.reshape(-1, channels)
    return np.mean(frames, axis=1)


def load_wav(path: str, minimum_duration: float = MIN_AUDIO_DURATION_SECONDS) -> AudioSignal:
    """Carga un archivo WAV PCM y devuelve su representación mono en punto flotante.

    El requisito predeterminado del proyecto es una duración mínima de 20 segundos.
    Se puede pasar ``minimum_duration=0`` cuando se usa un archivo corto como prueba.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"audio file not found: {file_path}")

    minimum_seconds = float(minimum_duration)
    if not np.isfinite(minimum_seconds) or minimum_seconds < 0.0:
        raise ValueError("minimum_duration must be a finite non-negative number")

    try:
        with wave.open(str(file_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()
            raw_data = wav_file.readframes(frame_count)
    except wave.Error as error:
        raise ValueError(f"invalid WAV file: {file_path}") from error

    if compression != "NONE":
        raise ValueError("compressed WAV files are not supported; use PCM WAV")
    if sample_rate <= 0:
        raise ValueError("the WAV file has an invalid sample rate")
    if sample_width not in SUPPORTED_SAMPLE_WIDTHS:
        raise ValueError("only 8, 16, 24, and 32-bit PCM WAV files are supported")

    interleaved = _decode_pcm(raw_data, sample_width)
    samples = _to_mono(interleaved, channels)
    duration = samples.size / sample_rate
    if duration < minimum_seconds:
        raise ValueError(
            f"audio must be at least {minimum_seconds:g} seconds long; "
            f"received {duration:.3f} seconds"
        )
    return AudioSignal(samples, sample_rate, channels, sample_width, str(file_path))


def resample_audio(audio: AudioSignal, target_rate: float) -> AudioSignal:
    """Devuelve una señal de audio remuestreada manualmente a ``target_rate``."""

    validated_rate = validate_sampling_frequency(target_rate)
    integer_rate = int(round(validated_rate))
    if not np.isclose(validated_rate, integer_rate):
        raise ValueError("audio sample rate must be an integer number of Hz")
    samples = resample_linear(audio.samples, audio.sample_rate, integer_rate)
    return AudioSignal(
        samples=samples,
        sample_rate=integer_rate,
        channels=1,
        sample_width=audio.sample_width,
        path=audio.path,
    )


def requantize_audio(audio: AudioSignal, levels: int) -> np.ndarray:
    """Cuantiza las muestras de audio y devuelve solo los valores reconstruidos."""

    from .dsp_core import quantize_uniform

    quantized = quantize_uniform(audio.samples, levels, -1.0, 1.0)
    return np.clip(quantized.values, -1.0, 1.0)


def play_audio(samples: np.ndarray, sample_rate: float, blocking: bool = False) -> None:
    """Reproduce muestras mono en punto flotante usando sounddevice."""

    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples contains non-finite values")
    rate = validate_sampling_frequency(sample_rate)

    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "sounddevice is required for audio playback; install it before playing"
        ) from error

    sd.play(np.clip(values, -1.0, 1.0), int(round(rate)), blocking=blocking)


def stop_audio() -> None:
    """Detiene cualquier audio que esté reproduciéndose con sounddevice."""

    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "sounddevice is required for audio playback; install it before stopping"
        ) from error
    sd.stop()


__all__ = [
    "AudioSignal",
    "MIN_AUDIO_DURATION_SECONDS",
    "load_wav",
    "play_audio",
    "requantize_audio",
    "resample_audio",
    "stop_audio",
]
