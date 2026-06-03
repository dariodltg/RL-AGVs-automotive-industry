# Plan de experimentación — Estado y diseño

Documento de contexto para redacción del capítulo de evaluación del TFM
("PPO vs A* para gestión de flotas de AGVs"). Resume las decisiones de
diseño tomadas, qué scripts existen, y qué falta por ejecutar.

Fecha del estado: 2026-05-31.

---

## 1. Estructura general — 4 experimentos

La evaluación se divide en cuatro experimentos independientes, cada uno
respondiendo a una pregunta concreta. Esto evita confundir "qué configuración
del algoritmo es buena" con "cómo escala el algoritmo frente al baseline".

| # | Experimento     | Pregunta                                                   | Estado      |
|---|-----------------|------------------------------------------------------------|-------------|
| 1 | HP search       | ¿Qué configuración de PPO funciona mejor en este problema? | Listo, sin ejecutar |
| 2 | Scaling         | ¿Cómo escala PPO vs A* vs random con el tamaño de flota?   | Listo, sin ejecutar (necesita ganador de exp. 1) |
| 3 | Robustness      | ¿Aguanta PPO cambios en la carga de tareas (zero-shot)?    | Listo, sin ejecutar (necesita ganador de exp. 1) |
| 4 | Layout          | ¿Generaliza PPO entre distintos layouts de planta?         | Futuro — solo existe el layout L1 |

El experimento 4 queda como **línea de continuación** en la memoria, junto a
la futura estación de Quality Inspection.

---

## 2. Decisiones de diseño tomadas

### 2.1 Hiperparámetros que se varían en el experimento 1

Solo se varían los tres parámetros con mayor impacto demostrado en PPO sobre
espacios de acción discretos y horizontes largos:

| Hiperparámetro  | Valores             | Justificación |
|-----------------|---------------------|---------------|
| `learning_rate` | 1e-4, 3e-4, 1e-3    | El más sensible. Sweep clásico (Schulman et al.). 3e-4 es default robusto; 1e-4 más estable; 1e-3 más agresivo. |
| `ent_coef`      | 0.0, 0.01, 0.05     | Crítico en routing: sin entropía PPO colapsa a política greedy subóptima. Queremos comprobar explícitamente que 0.0 degrada. |
| `n_steps`       | 1024, 2048          | Tamaño del rollout antes de cada update. Afecta varianza vs frecuencia de update; con episodios de 500 steps tiene impacto directo. |

**Total: 3 × 3 × 2 = 18 combinaciones**.

Hiperparámetros **fijos** en defaults de SB3 (registrados en cada sidecar JSON):
`batch_size=64`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`,
`clip_range=0.2`, `vf_coef=0.5`, `max_grad_norm=0.5`.

Hiperparámetros **descartados** y por qué:
- `gamma`: con horizon 500, 0.99 ya da effective horizon ~100 que cubre la
  tarea. Variarlo es ruido.
- `clip_range`: en PPO casi nunca importa; 0.2 es el "no lo toques".
- `batch_size` / `n_epochs`: efectos de segundo orden, no justifican
  multiplicar el coste de 18 combos por 4 o más.

### 2.2 `n_agvs` no es un hiperparámetro, es una condición del experimento

`n_agvs` cambia el problema (obs/action spaces dependen de él), no el
algoritmo. Por eso vive en el experimento 2 (scaling) y NO en la grid de
hiperparámetros. Si estuviera en la grid: 72 combos × 200k = ~14M timesteps,
prohibitivo, y se conflictarían las conclusiones "qué HP" con "qué fleet
size".

### 2.3 Robustness — modo zero-shot (no especialista)

Se entrena UN solo PPO a `load_level=medium` y se evalúa en `low/medium/high`.
Mide generalización fuera de la distribución de entrenamiento, que es el
caso realista en una planta donde la carga varía a lo largo del día. La
alternativa (especialista — un PPO por cada nivel de carga) es más fácil
estadísticamente pero menos interesante narrativamente para un TFM.

### 2.4 Baselines (A* + greedy, random)

A* y random **NO se entrenan**, solo se evalúan. Aparecen en los
experimentos 2, 3 y 4 — uno por cada condición experimental (n_agvs,
load_level, layout). El experimento 1 NO los necesita porque ahí estamos
comparando configuraciones de PPO entre sí.

---

## 3. Almacenamiento de resultados

### 3.1 SQLite — `C:\agv-experiments\experiments.db`

Una sola DB para los tres experimentos. Cada fila lleva un campo
`hyperparameters` JSON con un sub-campo `experiment` para distinguirlas.

Tablas:

- **`experiments`** — una fila por run (combinación agent × condición × seed):
  - `run_id`, `agent`, `seed`, `n_agvs`, `n_tasks_max`, `max_steps`,
    `task_arrival_rate`, `hyperparameters` (JSON), `status`, timestamps.
- **`episodes`** — una fila por episodio dentro de cada run:
  - `total_reward`, `tasks_completed`, `tasks_s0–s3` (por stage),
    `collisions`, `avg_cycle_time`, `mean_utilization`.

### 3.2 TensorBoard — `logs/tensorboard/<run_name>/`

Una subcarpeta por entrenamiento. Captura curvas de loss, mean_reward,
entropy, KL, FPS, etc. **Solo durante el entrenamiento**; no se usa para
evaluación. Comando: `tensorboard --logdir logs/tensorboard`.

### 3.3 Modelos en disco

- **Experimento 1**: `models/grid_hp/<combo_name>.zip` + sidecar `.json`
  con todos los hiperparámetros del combo.
- **Experimento 2**: `models/grid_scaling/ppo_scaling_n{N}.zip` (uno por
  tamaño de flota).
- **Experimento 3**: `models/grid_robustness/ppo_robustness.zip` (uno solo,
  entrenado a load=medium).

Cada `.zip` viene con un `.json` sidecar que contiene `hyperparameters`,
`env_config`, `timesteps`, `train_seed`, y la etiqueta `experiment`.

### 3.4 Por qué no se duplica el entrenamiento en SQLite

El entrenamiento por dentro (curvas de loss, KL, etc.) lo cubre TensorBoard
de sobra. En SQLite solo van los resultados de **evaluación final** que son
los que de verdad se comparan en el TFM (reward medio, tasks completados,
collisions, utilization) y los que Grafana visualiza.

---

## 4. Estado de la implementación

### 4.1 Scripts de ejecución

| Script                       | Experimento | Descripción |
|------------------------------|-------------|-------------|
| `run_grid_hp.py`             | 1           | 18 combos, train + eval headless, resumible |
| `run_grid_scaling.py`        | 2           | 4 trains (n_agvs ∈ {2,4,6,8}) + eval de 3 agentes en cada uno |
| `run_grid_robustness.py`     | 3           | 1 train @ medium + eval de 3 agentes en {low, medium, high} |
| `scripts/clean_registry.py`  | utilidad    | Borra filas de `experiments.db` con filtros (`--agent`, `--status`, `--before`, ...) |
| `scripts/clean_tensorboard.py` | utilidad  | Borra carpetas de TensorBoard con filtros (`--pattern`, `--older-than`, ...) |

Todos los scripts son headless (sin Pygame ni CoppeliaSim), salvan
checkpoint mínimo, son **resumibles** (si el `.zip` ya existe, saltan el
train de esa combinación), y siguen manejo de errores con captura por
combo: un fallo en un combo no aborta el resto del grid.

### 4.2 Workflow recomendado

```
1. python run_grid_hp.py                       # ~3.5h, 18 combos
2. Identificar combo ganador (Grafana o SQL)
3. Editar WINNING_HP en run_grid_scaling.py y run_grid_robustness.py
4. python run_grid_scaling.py                  # ~1h
5. python run_grid_robustness.py               # ~20 min
6. Generar gráficas comparativas desde Grafana
```

Estimaciones derivadas del smoke test: 5000 timesteps tardan ~17s en la
máquina actual (sin GPU), extrapolación lineal a 200000 steps.

### 4.3 Cambios al código del agente

`PPOAgent.__init__` acepta `**ppo_kwargs` que se reenvían directamente al
constructor de SB3 PPO. No requirió cambios para soportar la grid: los
hiperparámetros del combo se pasan como kwargs.

`PPOAgent.train` acepta `extra_callbacks` (añadido durante el desarrollo
del modo Training del launcher, reutilizable por los runners si se quisiera
tracking de progreso).

---

## 5. Filtros útiles en Grafana / SQL

Como los tres experimentos comparten DB, se distinguen por el campo
`hyperparameters.experiment`:

```sql
-- Solo experimento 1 (HP search)
WHERE json_extract(hyperparameters, '$.grid_combo') IS NOT NULL

-- Solo experimento 2 (scaling)
WHERE json_extract(hyperparameters, '$.experiment') = 'scaling'

-- Solo experimento 3 (robustness)
WHERE json_extract(hyperparameters, '$.experiment') = 'robustness'

-- Mejor combo de HP search (top 5 por reward medio)
SELECT json_extract(e.hyperparameters, '$.grid_combo') AS combo,
       ROUND(AVG(ep.total_reward), 2) AS mean_reward
FROM   experiments e
JOIN   episodes    ep USING (run_id)
WHERE  e.agent  = 'ppo'
  AND  e.status = 'completed'
  AND  json_extract(e.hyperparameters, '$.grid_combo') IS NOT NULL
GROUP  BY combo
ORDER  BY mean_reward DESC
LIMIT  5;

-- Robustness: degradación de PPO al pasar de medium (train) a high (eval)
SELECT json_extract(e.hyperparameters, '$.eval_load_level') AS load,
       ROUND(AVG(ep.total_reward), 2) AS mean_reward
FROM   experiments e
JOIN   episodes    ep USING (run_id)
WHERE  e.agent = 'ppo'
  AND  json_extract(e.hyperparameters, '$.experiment') = 'robustness'
GROUP  BY load;
```

---

## 6. Métricas que se reportan en cada run

Por episodio (tabla `episodes`):
- `total_reward` — suma de recompensas del episodio.
- `tasks_completed` — tareas entregadas (todos los stages combinados).
- `tasks_s0..s3` — desglose por stage del pipeline de producción.
- `collisions` — número de colisiones evitadas / detectadas.
- `avg_cycle_time` — tiempo medio en pasos entre que una tarea se asigna
  y se completa.
- `mean_utilization` — fracción de pasos en que los AGVs están con tarea
  activa.

Por run (tabla `experiments`):
- Identificación: `run_id`, `agent`, `seed`, `hyperparameters` JSON.
- Condiciones: `n_agvs`, `n_tasks_max`, `max_steps`, `task_arrival_rate`.
- Estado: `status` (`running`/`completed`/`interrupted`), timestamps.

El método `ExperimentRegistry.summary()` imprime un resumen agregado por
run_id en stdout al final de cada script.

---

## 7. Pendientes inmediatos

1. **Ejecutar el experimento 1** (`python run_grid_hp.py`), ~3.5h.
2. **Identificar el combo ganador** con la query SQL del bloque 5 o desde
   Grafana.
3. **Editar `WINNING_HP`** en `run_grid_scaling.py:32-34` y
   `run_grid_robustness.py:53-55`.
4. **Ejecutar experimentos 2 y 3**.
5. **Sección de la memoria**: redactar capítulo de evaluación usando las
   queries de la sección 5, exportando gráficas desde Grafana.

---

## 8. Lo que NO se ha hecho (y por qué)

- **Paralelización de entrenamientos**: PPO en CPU no se beneficia de
  paralelizar varios procesos por contención de GIL/numpy; secuencial es
  suficiente para 18 trains de 200k cada uno.
- **Sweep tipo random search / Sobol**: la grid completa de 18 combos cabe
  en presupuesto razonable y es más interpretable para el TFM que un sweep
  aleatorio.
- **Train con múltiples seeds por combo**: una sola seed de train por combo
  para mantener el coste bajo. La robustez estadística viene de las 5 seeds
  × 20 episodios de evaluación (= 100 datos por combo).
- **Schema de SQLite para curvas de entrenamiento**: las cubre TensorBoard
  sin duplicación.

---

## 9. Referencia rápida de rutas

- DB: `C:\agv-experiments\experiments.db`
- TensorBoard: `logs/tensorboard/<run_name>/`
- Modelos: `models/grid_hp/`, `models/grid_scaling/`, `models/grid_robustness/`
- Sidecars JSON: junto a cada `.zip`
- Scripts de utilidad: `scripts/clean_registry.py`, `scripts/clean_tensorboard.py`
