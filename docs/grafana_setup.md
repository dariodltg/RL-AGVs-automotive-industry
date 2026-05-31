# Grafana setup — RL-AGVs experiment dashboard

Grafana reads `experiments.db` (SQLite, WAL mode) in **read-only** mode through the
community plugin **frser-sqlite-datasource**. No data is written from Grafana.

---

## 1. Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Grafana | ≥ 10.x | OSS or Enterprise |
| frser-sqlite-datasource | latest | `grafana-cli plugins install frser-sqlite-datasource` |

---

## 2. Install the SQLite plugin

```bash
grafana-cli plugins install frser-sqlite-datasource
# then restart Grafana
```

On Docker:

```yaml
environment:
  GF_INSTALL_PLUGINS: frser-sqlite-datasource
```

---

## 3. Add the datasource

1. Open **Connections → Data sources → Add new data source**.
2. Search for **SQLite** and select it.
3. Set **Path** to the absolute path of `experiments.db`, e.g.:
   ```
   /home/user/RL-AGVs-automotive-industry/experiments.db
   ```
4. Enable **Read-only mode** (prevents accidental writes).
5. Click **Save & Test** — you should see "Database Connected".

> On Windows use a forward-slash path or double backslashes:
> `C:/Users/dario/RL-AGVs-automotive-industry/experiments.db`

---

## 4. SQL queries per panel

### 4.1 Run summary table

Aggregated view equivalent to `ExperimentRegistry.summary()`.

```sql
SELECT
    e.run_id,
    e.agent,
    e.seed,
    e.status,
    ROUND(AVG(ep.total_reward), 2)     AS mean_reward,
    ROUND(AVG(ep.tasks_completed), 2)  AS mean_tasks,
    ROUND(AVG(ep.collisions), 2)       AS mean_collisions,
    ROUND(AVG(ep.mean_utilization), 4) AS mean_util,
    COUNT(ep.episode)                  AS n_episodes
FROM experiments e
JOIN episodes ep USING (run_id)
WHERE e.status = 'completed'
GROUP BY e.run_id
ORDER BY e.agent, e.seed
```

Panel type: **Table**.

---

### 4.2 Mean reward per agent (bar chart)

```sql
SELECT
    e.agent,
    ROUND(AVG(ep.total_reward), 2) AS mean_reward
FROM experiments e
JOIN episodes ep USING (run_id)
WHERE e.status = 'completed'
GROUP BY e.agent
ORDER BY mean_reward DESC
```

Panel type: **Bar chart** — X: `agent`, Y: `mean_reward`.

---

### 4.3 Tasks completed per stage (stacked bars)

```sql
SELECT
    e.agent,
    ROUND(AVG(ep.tasks_s0), 2) AS stage_0,
    ROUND(AVG(ep.tasks_s1), 2) AS stage_1,
    ROUND(AVG(ep.tasks_s2), 2) AS stage_2,
    ROUND(AVG(ep.tasks_s3), 2) AS stage_3
FROM experiments e
JOIN episodes ep USING (run_id)
WHERE e.status = 'completed'
GROUP BY e.agent
```

Panel type: **Bar chart** — stacked mode, one series per stage column.

---

### 4.4 Mean collisions per run

```sql
SELECT
    e.run_id,
    e.agent,
    e.seed,
    ROUND(AVG(ep.collisions), 2) AS mean_collisions
FROM experiments e
JOIN episodes ep USING (run_id)
WHERE e.status = 'completed'
GROUP BY e.run_id
ORDER BY e.agent, e.seed
```

Panel type: **Bar chart**.

---

### 4.5 Fleet utilisation (%)

```sql
SELECT
    e.agent,
    ROUND(AVG(ep.mean_utilization) * 100, 2) AS utilization_pct
FROM experiments e
JOIN episodes ep USING (run_id)
WHERE e.status = 'completed'
GROUP BY e.agent
```

Panel type: **Bar gauge** — unit: `percent (0-100)`.

---

### 4.6 Mean cycle time

```sql
SELECT
    e.agent,
    ROUND(AVG(ep.avg_cycle_time), 2) AS mean_cycle_time
FROM experiments e
JOIN episodes ep USING (run_id)
WHERE e.status = 'completed'
GROUP BY e.agent
```

Panel type: **Bar chart** — unit: `steps`.

---

### 4.7 Reward per episode within a run (time series)

Useful to inspect stability of a single run.  Use a Grafana variable
`$run_id` (type: Query, query: `SELECT run_id FROM experiments`) to
make the panel interactive.

```sql
SELECT
    ep.episode,
    ep.total_reward,
    e.agent
FROM episodes ep
JOIN experiments e USING (run_id)
WHERE ep.run_id = '$run_id'
ORDER BY ep.episode
```

Panel type: **Time series** (treat `episode` as the X axis; in the
plugin map `episode` → X field).

---

### 4.8 Reward variance across seeds (same agent)

```sql
SELECT
    e.agent,
    e.seed,
    ROUND(AVG(ep.total_reward), 2) AS mean_reward,
    ROUND(
        SQRT(AVG(ep.total_reward * ep.total_reward) -
             AVG(ep.total_reward) * AVG(ep.total_reward)), 2
    ) AS std_reward
FROM experiments e
JOIN episodes ep USING (run_id)
WHERE e.status = 'completed'
GROUP BY e.agent, e.seed
ORDER BY e.agent, e.seed
```

Panel type: **Table** or **Bar chart** grouped by `agent`.

---

## 5. Importable dashboard JSON

Save the block below as `agv_dashboard.json` and import it via
**Dashboards → Import → Upload JSON file**. Replace `<DATASOURCE_UID>`
with the UID shown in your datasource settings page.

```json
{
  "__inputs": [
    {
      "name": "DS_SQLITE",
      "label": "SQLite experiments",
      "type": "datasource",
      "pluginId": "frser-sqlite-datasource"
    }
  ],
  "__requires": [
    { "type": "grafana", "id": "grafana", "name": "Grafana", "version": "10.0.0" },
    { "type": "datasource", "id": "frser-sqlite-datasource", "name": "SQLite", "version": "3.0.0" }
  ],
  "title": "RL-AGVs — Experiment Comparison",
  "uid": "agv-rl-tfm",
  "schemaVersion": 38,
  "version": 1,
  "panels": [
    {
      "id": 1,
      "title": "Run summary",
      "type": "table",
      "gridPos": { "h": 8, "w": 24, "x": 0, "y": 0 },
      "datasource": { "type": "frser-sqlite-datasource", "uid": "${DS_SQLITE}" },
      "targets": [
        {
          "rawQueryText": "SELECT e.run_id, e.agent, e.seed, e.status, ROUND(AVG(ep.total_reward),2) AS mean_reward, ROUND(AVG(ep.tasks_completed),2) AS mean_tasks, ROUND(AVG(ep.collisions),2) AS mean_collisions, ROUND(AVG(ep.mean_utilization),4) AS mean_util, COUNT(ep.episode) AS n_episodes FROM experiments e JOIN episodes ep USING (run_id) WHERE e.status='completed' GROUP BY e.run_id ORDER BY e.agent, e.seed",
          "queryType": "table",
          "refId": "A"
        }
      ]
    },
    {
      "id": 2,
      "title": "Mean reward per agent",
      "type": "barchart",
      "gridPos": { "h": 8, "w": 8, "x": 0, "y": 8 },
      "datasource": { "type": "frser-sqlite-datasource", "uid": "${DS_SQLITE}" },
      "targets": [
        {
          "rawQueryText": "SELECT e.agent, ROUND(AVG(ep.total_reward),2) AS mean_reward FROM experiments e JOIN episodes ep USING (run_id) WHERE e.status='completed' GROUP BY e.agent ORDER BY mean_reward DESC",
          "queryType": "table",
          "refId": "A"
        }
      ]
    },
    {
      "id": 3,
      "title": "Fleet utilisation (%)",
      "type": "bargauge",
      "gridPos": { "h": 8, "w": 8, "x": 8, "y": 8 },
      "datasource": { "type": "frser-sqlite-datasource", "uid": "${DS_SQLITE}" },
      "options": { "reduceOptions": { "calcs": ["lastNotNull"] }, "orientation": "horizontal" },
      "fieldConfig": { "defaults": { "unit": "percent", "min": 0, "max": 100 } },
      "targets": [
        {
          "rawQueryText": "SELECT e.agent, ROUND(AVG(ep.mean_utilization)*100,2) AS utilization_pct FROM experiments e JOIN episodes ep USING (run_id) WHERE e.status='completed' GROUP BY e.agent",
          "queryType": "table",
          "refId": "A"
        }
      ]
    },
    {
      "id": 4,
      "title": "Mean collisions per agent",
      "type": "barchart",
      "gridPos": { "h": 8, "w": 8, "x": 16, "y": 8 },
      "datasource": { "type": "frser-sqlite-datasource", "uid": "${DS_SQLITE}" },
      "targets": [
        {
          "rawQueryText": "SELECT e.agent, ROUND(AVG(ep.collisions),2) AS mean_collisions FROM experiments e JOIN episodes ep USING (run_id) WHERE e.status='completed' GROUP BY e.agent",
          "queryType": "table",
          "refId": "A"
        }
      ]
    },
    {
      "id": 5,
      "title": "Tasks per stage (stacked)",
      "type": "barchart",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 16 },
      "datasource": { "type": "frser-sqlite-datasource", "uid": "${DS_SQLITE}" },
      "options": { "stacking": "normal" },
      "targets": [
        {
          "rawQueryText": "SELECT e.agent, ROUND(AVG(ep.tasks_s0),2) AS stage_0, ROUND(AVG(ep.tasks_s1),2) AS stage_1, ROUND(AVG(ep.tasks_s2),2) AS stage_2, ROUND(AVG(ep.tasks_s3),2) AS stage_3 FROM experiments e JOIN episodes ep USING (run_id) WHERE e.status='completed' GROUP BY e.agent",
          "queryType": "table",
          "refId": "A"
        }
      ]
    },
    {
      "id": 6,
      "title": "Mean cycle time per agent",
      "type": "barchart",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 16 },
      "datasource": { "type": "frser-sqlite-datasource", "uid": "${DS_SQLITE}" },
      "targets": [
        {
          "rawQueryText": "SELECT e.agent, ROUND(AVG(ep.avg_cycle_time),2) AS mean_cycle_time FROM experiments e JOIN episodes ep USING (run_id) WHERE e.status='completed' GROUP BY e.agent",
          "queryType": "table",
          "refId": "A"
        }
      ]
    }
  ]
}
```

---

## 6. Workflow

```
# 1. Train PPO (writes models/ppo_fleet.zip + models/ppo_fleet.json)
python train_ppo.py --run_name ppo_fleet --timesteps 300000

# 2. Evaluate all agents with the same seeds
python scripts/evaluate.py --agent astar  --seeds 0 1 2 3 4
python scripts/evaluate.py --agent random --seeds 0 1 2 3 4
python scripts/evaluate.py --agent ppo    --model models/ppo_fleet.zip --seeds 0 1 2 3 4

# 3. Quick sanity check (CLI)
python -c "from src.experiments import ExperimentRegistry; ExperimentRegistry().summary()"

# 4. Open Grafana dashboard and refresh
```

Training diagnostics (loss, `approx_kl`, `clip_fraction`, `explained_variance`,
`rollout/ep_rew_mean`) are in TensorBoard, not in Grafana:

```bash
tensorboard --logdir logs/tensorboard
```
