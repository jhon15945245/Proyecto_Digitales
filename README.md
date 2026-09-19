
#Grupo: Alexander Bonilla Higidio - Jhonatan Estiven Gurrute
# Simulación de un sistema de digitalización de señales
Proyecto de Comunicaciones Digitales - Universidad del Cauca.

## Requisitos

- Python 3.10 o superior.
- NumPy.
- Matplotlib.
- sounddevice (solo necesario para reproducir audio).

## Instalación

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Desde la carpeta raíz del proyecto:

```bash
python Main.py
```

## Funcionalidades

- Señales sintéticas A, B y C según la guía.
- Carga de señal D (WAV) con mínimo 20 s.
- Frecuencia de muestreo configurable.
- Cálculo de Ts = 1/fs.
- Muestreo mediante instantes x(nTs).
- Cuantificador uniforme con 2, 4, 8, 16, 32, 64, 128 o 256 niveles.
- Cálculo de paso de cuantificación Delta.
- MSE y SQNR de cuantificación.
- Error espectral relativo entre la señal original y la reconstruida.
- Reconstrucción mediante retención de orden cero (ZOH).
- Resampling manual por interpolación lineal para audio, sin scipy.signal.
- Reproducción del audio original y reconstruido.
- Exportación de Tabla 1 y Tabla 2 en CSV y PNG.

## Experimentos de la sustentación

El botón **Exportar tablas** genera:

- `tabla_1_cuantificacion.csv`
- `tabla_1_cuantificacion.png`
- `tabla_2_frecuencia_muestreo.csv`
- `tabla_2_frecuencia_muestreo.png`

La Tabla 1 usa la señal C, `a = 5 Hz` y `fs = 10 Hz`, correspondiente a la frecuencia de Nyquist de la señal C.

La Tabla 2 usa el audio cargado, 16 niveles y los factores indicados en la guía:
0.25fs, 0.5fs, 0.75fs, fs, 1.25fs, 1.5fs, 1.75fs y 2fs.

## Nota sobre la referencia PCM

El archivo WAV ya es una señal digital. Por ello, no se dispone de la señal analógica previa a su cuantificación original. La aplicación muestra una referencia reproducible basada en la resolución nominal del PCM del WAV (por ejemplo, 16 bits = 65536 niveles), usando un cuantificador uniforme sobre [-1, 1].
