# Detección de Comunidades en Redes — Álgebra Lineal Computacional

**Trabajo Práctico II — Licenciatura en Ciencias de Datos, UBA (FCEyN) — 2025**

## Descripción

Análisis de la estructura de comunidades en una red de museos de CABA, aplicando
métodos de partición de grafos basados en álgebra lineal.

## Métodos implementados

- **Método del Laplaciano**: partición recursiva minimizando cortes entre comunidades,
  usando el segundo autovector más pequeño de la matriz Laplaciana (L = K − A).
- **Método de Modularidad**: partición optimizando la densidad interna de comunidades
  usando el autovector dominante de la matriz de modularidad (R).
- **Método de la Potencia e Inverso**: cálculo iterativo de autovectores dominantes
  y secundarios.
- **Shifting de autovalores y Deflación de Hotelling**: extracción de autovectores
  sucesivos sin recalcular desde cero.
- **Descomposición LU**: resolución eficiente de sistemas lineales.

## Tecnologías

Python · NumPy · SciPy · NetworkX · GeoPandas · Matplotlib

## Estructura

- `TP2.ipynb` — Notebook principal con desarrollo teórico, demostraciones y experimentos
- `funciones.py` — Implementaciones propias de los algoritmos
