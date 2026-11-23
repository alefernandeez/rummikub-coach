# Experimento con el dataset de KU Leuven

Dataset: Vandevelde, Lauwers, Vennekens (2026), *Rummikub Images Annotated Dataset*, KU Leuven RDR,
https://doi.org/10.48804/AXW1W4 (CC-BY-4.0). Artículo: *Enhancing Computer Vision with Knowledge: a Rummikub Case Study*, ESANN 2025 (arXiv:2411.18172).

Los datos no están en este repositorio: hay que descargarlos y reducirlos a 1600 px (`manifest.json` guarda la escala).

| Modelo | Detectadas (test) | Bien leídas / total (test) | Tu foto: fichas bien leídas (de 53) |
|---|---|---|---|
| Original (solo sintético) | 84,3 % | 71,6 % | 35 (38 con corrección) |
| Afinado con KU Leuven | 96,1 % | 82,4 % | 31 |

Conclusión: sus fichas tienen otro diseño (número centrado en un círculo, logo abajo) y otros fondos, así que
afinar con ellas mejora su test pero empeora con nuestras fichas. Lo que sí se transfiere es la
**corrección por reglas** del artículo (`correctSet` en `vision.js`): elegir la lectura válida más probable de cada set.
