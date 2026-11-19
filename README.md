# Rummikub Coach

Web app para iPhone que **lee las fichas con la cámara** (modelo propio que funciona sin conexión, dentro del navegador) y **calcula la jugada que coloca más fichas** con las reglas oficiales de Rummikub.

**App:** https://alefernandeez.github.io/rummikub-coach/ — ábrela en Safari, *Compartir → Añadir a pantalla de inicio*. Después de la primera visita funciona sin internet.

## Cómo se usa
1. **Activar cámara** (Safari pide permiso una vez).
2. Apunta a tu atril → **Escanear atril**. Apunta a la mesa → **Escanear mesa**. La app toma el fotograma más nítido de ~0,5 s.
3. Revisa lo leído: borde naranja discontinuo = ficha dudosa, set en rojo = set inválido. Toca cualquier ficha para corregirla.
4. **Calcular mejor jugada.** Si se pueden colocar las mismas fichas sin romper ningún set, la app propone esa jugada; si no, enseña la jugada máxima y los sets que cambian, y ofrece la alternativa sin reorganizar.
5. **He hecho esta jugada** actualiza tu atril y la mesa.

## Cómo funciona
| Pieza | Qué hace |
|---|---|
| `solver.js` | Programación dinámica exacta (número × color) que maximiza las fichas colocadas reorganizando la mesa. Tiene en cuenta comodines, la primera bajada de 30 puntos solo con fichas propias y grupos de hasta 6 colores. Verificada contra un ILP (HiGHS) en más de 1.000 partidas aleatorias sin ninguna diferencia. |
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
