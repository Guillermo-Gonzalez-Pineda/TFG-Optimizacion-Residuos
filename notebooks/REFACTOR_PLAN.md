# Refactor de cuadernos — notas

## Política de outputs en cuadernos

Los cuadernos de `notebooks/` se versionan **con sus outputs ejecutados**
(tablas e imágenes embebidas), para que sean **autocontenidos**: el tribunal ve
los resultados sin ejecutar nada.

Reglas al commitear un cuaderno:

- **Se ejecuta con kernel limpio** antes de commitear (`jupyter nbconvert --to
  notebook --execute --inplace`), para que los outputs correspondan al código
  actual y a la librería `analisis`.
- **Se limpia solo la metadata volátil**, no los outputs: `execution_count` a
  `null` y se elimina `metadata.execution` (timestamps por celda). Así el diff
  refleja cambios reales, no ruido de ejecución. **No usar `nbstripout`**, que
  borraría los outputs; usar un limpiador que preserve `outputs`.

Adopción por cuaderno (se aplica **al refactorizarse**, no antes):

- **01 — Validación de instancias**: refactorizado a "narrativa + llamadas" y
  versionado con outputs bajo esta política.
- **00 / 02 / 03**: siguen **sin outputs** a propósito hasta su refactor; adoptan
  esta política cuando se reescriban.
- **03 — Relajación lagrangiana**: además, **hoy no ejecuta** (lee campos
  inexistentes `best_feasible` / `runtime_seconds`); solo se versionará con
  outputs una vez arreglado.
