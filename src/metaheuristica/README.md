# Metaheurística (búsqueda tabú) — compilación

Build reproducible con **CMake** (C++20 exigido). Reemplaza la compilación a mano.

- El **compilador NO se fija** en el `CMakeLists.txt`: se elige al configurar, por
  autodetección o con `-DCMAKE_CXX_COMPILER=...` (g++-13, g++ 15.2, clang++…).
- Dos build types:
  - **Release**: `-O2`, **sin** oráculo → binario de producción.
  - **Debug**: `-O2` con `-DORACLE_CHECK` → binario con el juez de consistencia.
    Debug es *"modo oráculo"*, **no** depuración con gdb: el `-O2` es a propósito
    (el oráculo es IEEE-estable a `-O2` y ~7× más rápido que a `-O0`).

Todos los comandos se ejecutan desde `src/metaheuristica/`. `build/` está en
`.gitignore`, así que los artefactos no se versionan.

## (a) Release — autodetección de compilador

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
# → build/release/metaheuristica
```

## (b) Release — fijando g++-13 (reproducibilidad para la memoria)

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-13
cmake --build build/release -j
```

## (c) Debug — con oráculo

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j
# → build/debug/metaheuristica  (con el juez de consistencia activo)
```

## Ejecución

```bash
# Modo (defecto per-tipo) + ruta de instancia (argumento posicional):
./build/release/metaheuristica --modo=entero   ../../data/processed/instancia_laguna_500m.json
./build/release/metaheuristica --modo=per-tipo  ../../data/processed/instancia_laguna_1000m.json
```

En Debug, el oráculo verifica cada movimiento y aborta con diagnóstico si detecta
una inconsistencia; en Release no hay ni rastro del oráculo en el binario.
