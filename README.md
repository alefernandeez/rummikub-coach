# Rummikub Coach

Web app para iPhone que **lee las fichas con la cámara** (modelo propio que funciona sin conexión, dentro del navegador) y **calcula la jugada que coloca más fichas** con las reglas oficiales de Rummikub.

**App:** https://alefernandeez.github.io/rummikub-coach/ — ábrela en Safari, *Compartir → Añadir a pantalla de inicio*. Después de la primera visita funciona sin internet.

## Cómo se usa
1. **▶ Empezar a escanear** (Safari pide permiso de cámara la primera vez).
2. Con **Enfoco mi atril** seleccionado, apunta a tu atril hasta que todas sus fichas estén en verde.
3. Cambia a **Enfoco la mesa** y recórrela despacio. **Verde** = ficha fijada (leída igual en varios fotogramas); **naranja** = acércate a esa ficha hasta que se ponga verde. Los contadores de arriba dicen cuántas quedan.
4. **■ Terminar y calcular**: se apaga la cámara y aparece la mejor jugada. Puedes tocar cualquier ficha para corregirla y volver a calcular.
5. **He hecho esta jugada** actualiza tu atril y la mesa.

## Cómo funciona
| Pieza | Qué hace |
|---|---|
| `solver.js` | Programación dinámica exacta (número × color) que maximiza las fichas colocadas reorganizando la mesa. Tiene en cuenta comodines, la primera bajada de 30 puntos solo con fichas propias y grupos de hasta 6 colores. Verificada contra un ILP (HiGHS) en más de 1.000 partidas aleatorias sin ninguna diferencia. |
| `live.js` | Escaneo continuo: alinea cada fotograma con el mapa de fichas ya conocidas (RANSAC de similitud), acumula votos por ficha y la fija cuando varias lecturas coinciden. Las fichas demasiado pequeñas en la imagen no votan: se quedan en naranja hasta que te acercas. |
| `vision.js` | 1) **Detector** tipo CenterNet (mini U-Net, ~0,9 M de parámetros): predice el centro de cada ficha, su dirección "arriba" (la marca redonda va debajo del número) y su tamaño. 2) **Clasificador** (~0,4 M) sobre la ficha enderezada: número (1–13, comodín o "no es ficha"), color (6) y si está girada 180°. 3) Agrupa las fichas vecinas en sets. |
| `models/*.onnx` | Modelos ejecutados con onnxruntime-web (WASM), incluido en `vendor/`. |
| `sw.js` | Caché offline (service worker). |
| `ml/` | Generador de escenas sintéticas, entrenamiento (PyTorch) y exportación a ONNX. |

## Reentrenar con fotos reales
Cada escaneo y tus correcciones se guardan solo en el móvil (IndexedDB). En **⚙︎ Juego → Exportar datos de entrenamiento** se descargan como `.jsonl` (imagen + predicciones + etiquetas corregidas), para afinar los modelos con imágenes reales.

```bash
cd ml
python3 gen.py data/det_0.npz 1600 10   # escenas sintéticas
python3 train_det.py 6 1                # detector
python3 train_cls.py 8 1                # clasificador
python3 export.py && cp *.onnx ../models/
python3 infer.py foto.jpg 960 pred.png  # prueba con una foto
```
