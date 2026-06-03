# Integración con CoppeliaSim — Diseño y decisiones técnicas

Documento de contexto para redacción del capítulo de implementación del TFM
("PPO vs A* para gestión de flotas de AGVs"). Resume cómo se ha integrado
CoppeliaSim como visualizador 3D para el entorno Python, qué decisiones de
arquitectura se han tomado, y por qué.

Fecha del estado: 2026-05-31.

---

## 1. Objetivo de la integración

CoppeliaSim aporta una **capa de visualización 3D** sobre el entorno
Python existente. La motivación es triple:

1. **Validación visual**: comprobar que las políticas (PPO y A*) producen
   trayectorias razonables en un entorno con apariencia industrial real.
2. **Demostración**: figura del TFM y posibles presentaciones futuras
   con renderizado realista en lugar de la rejilla 2D de Pygame.
3. **Modelo de robot creíble**: usar el modelo `OmniPlatform`
   incluido en CoppeliaSim, con orientación de cuerpo según dirección
   de movimiento, en vez de cuadrados de colores.

**Lo que NO se hace**: CoppeliaSim **no participa en la simulación física
ni en el entrenamiento**. Es un visualizador puro. El entorno Gym
(`AGVFleetEnv`) es la única fuente de verdad sobre estado, recompensas y
transiciones.

---

## 2. Decisiones de arquitectura clave

### 2.1 Separación en dos componentes

Toda la integración vive en `src/coppeliasim/` y se divide en dos
módulos con responsabilidades claras:

- **`scene_builder.py`** — Construye la escena 3D **programáticamente**
  desde la rejilla `PlantMap` (20×20). Se ejecuta una sola vez (o
  cuando cambia el número de AGVs) y deja la escena lista para usar.
- **`bridge.py`** — Sincroniza estado en runtime. Cada paso del entorno
  empuja posiciones de AGVs, orientaciones de cuerpo, colores de
  estado y marcadores de tareas a CoppeliaSim.

Ambos están desacoplados: el scene_builder no sabe nada del bridge, y el
bridge no construye nada — solo resuelve handles ya existentes y los
manipula.

### 2.2 Bridge unidireccional Python → CoppeliaSim

El bridge solo escribe en CoppeliaSim, nunca lee. El bucle de
entrenamiento/evaluación no depende del visualizador, por lo que:

- Puede correr sin CoppeliaSim (modo `pygame` o `none`).
- Si CoppeliaSim se desconecta, el entorno sigue funcionando — solo
  deja de visualizarse.
- Garantiza reproducibilidad: la política nunca ve datos de CoppeliaSim.

### 2.3 Construcción programática de la escena

En lugar de mantener un `.ttt` (archivo de escena de CoppeliaSim) en el
repositorio, la escena se genera por código leyendo `PlantMap`. Ventajas:

- **Cero acoplamiento entre layout y CoppeliaSim**: si cambia la
  rejilla, basta con regenerar la escena.
- **No hay binarios en git**: solo código Python versionado.
- **Reproducible**: cualquiera que clone el repo y arranque CoppeliaSim
  vacío puede reconstruir la escena con un solo comando.
- **Limpieza automatizada**: `clear_scene()` borra todo lo creado
  por iteraciones anteriores antes de reconstruir, sin pasos manuales.

### 2.4 No se inicia la simulación física

La simulación de CoppeliaSim se mantiene en estado **stopped**. Razón:
los modelos OmniPlatform y similares incluyen scripts Lua internos
(controladores PID, integración de ruedas) que entran en conflicto con
las llamadas kinemáticas `setObjectPosition`. Si la simulación corriera,
los AGVs intentarían "obedecer" su control interno mientras el bridge los
"teletransporta" cada paso, provocando saltos visuales y errores.

Para visualización pura es innecesario: solo necesitamos que los modelos
se muevan a las posiciones que les indica el bridge.

---

## 3. Stack técnico

| Componente            | Tecnología / versión |
|-----------------------|----------------------|
| CoppeliaSim           | EDU 4.x (cualquier instalación con la API ZeroMQ) |
| Cliente Remote API    | `coppeliasim-zmqremoteapi-client` (PyPI) |
| Protocolo de transporte | ZeroMQ sobre TCP (puerto por defecto 23000) |
| Modelo de AGV         | `Omnidirectional platform.ttm` (incluido en `models/robots/mobile/` de CoppeliaSim) |
| Ejes                  | CoppeliaSim usa Z-up. La rejilla 20×20 mapea `(row, col)` a `(Y, X)` en metros |
| Tamaño de celda       | 0.5 m por celda → planta de 10 m × 10 m |

La elección de **ZeroMQ Remote API** sobre la antigua Legacy Remote API
es deliberada: la ZMQ es la API oficial moderna, sin necesidad de
plugins compilados, y expone toda la jerarquía de objetos vía
`sim.getObject('/path/to/obj')`.

---

## 4. `scene_builder.py` — Construcción de la escena

### 4.1 Mapeo `PlantMap` → objetos 3D

Cada celda de la rejilla (`CellType`) se traduce a un cuboide simple
con un color y una altura:

| CellType  | Color    | Altura | Significado físico |
|-----------|----------|--------|--------------------|
| FREE      | gris     | 0.03 m | Pasillo / suelo libre |
| OBSTACLE  | gris osc.| 1.5 m  | Muros, maquinaria fija |
| ENTRY     | verde    | 0.04 m | Punto de entrada de tareas |
| STAMPING  | azul     | 0.04 m | Estación de estampado |
| BUFFER    | naranja  | 0.04 m | Buffer intermedio |
| WELDING   | rojo     | 0.04 m | Estación de soldadura |
| EXIT      | amarillo | 0.04 m | Punto de salida |
| CHARGING  | cian     | 0.04 m | Estaciones de carga |

Las celdas se separan visualmente con un gap de 0.01 m entre ellas para
que se distinga la rejilla en la vista superior. Las estaciones son
tiles planos del mismo grosor que el suelo — solo OBSTACLE es alto,
representando paredes / maquinaria fija.

### 4.2 Jerarquía de la escena generada

```
Plant (dummy raíz)
├── Ground                      cuboide del plano de suelo
├── FREE_0_0, FREE_0_1, ...     400 tiles (uno por celda)
├── STAMPING_5_8, BUFFER_..., etc.
└── (...)

AGVs (dummy raíz)
├── AGV_0                       dummy de control
│   ├── <OmniPlatform model>    modelo .ttm cargado, hijo del dummy
│   └── AGV_0_Light             esfera del estado (color cambia en runtime)
├── AGV_1
└── ...
```

**Decisión clave: dummy neutro como raíz de cada AGV**. El modelo
OmniPlatform tiene una orientación natural propia que dificulta rotarlo
directamente. Usando un dummy con orientación `[0, 0, 0]` como padre,
el bridge puede aplicar `setObjectOrientation(dummy, -1, [0, 0, heading])`
sin componer ángulos de Euler — el modelo y la luz, como hijos del
dummy, rotan automáticamente preservando su pose local.

### 4.3 Carga del modelo OmniPlatform

Se resuelve la ruta dinámicamente con
`sim.getStringParam(sim.stringparam_application_path)` + `models/robots/mobile/`,
lo que evita hardcoded paths y funciona en cualquier instalación. Si el
modelo no existe se lanza `FileNotFoundError` con instrucciones claras.

### 4.4 Limpieza programática

`clear_scene()` recupera todos los handles via
`sim.getObjectsInTree(-1, -1, 0)` y los elimina en una sola llamada
`sim.removeObjects(handles, False)`. Si esta llamada batch falla (por
ejemplo, hay objetos no eliminables como la cámara default), se hace
fallback a eliminar uno a uno, ignorando los que fallen. Esto permite
que el usuario reconstruya la escena tantas veces como quiera sin
abrir el menú de CoppeliaSim.

Adicionalmente, el suelo por defecto de CoppeliaSim (`ResizableFloor_5_25`
o `Floor`) se elimina específicamente para no quedar superpuesto con el
suelo nuevo de la planta.

---

## 5. `bridge.py` — Sincronización en runtime

### 5.1 Conexión robusta con timeout

`_connect_zmq(host, port)` ejecuta el handshake bloqueante de
`RemoteAPIClient` en un **hilo daemon** con un `threading.Event` y un
timeout de 5 segundos. Si CoppeliaSim no está corriendo, en vez de
bloquear indefinidamente el lanzador, lanza `ConnectionError` con
instrucciones de qué hacer ("arranca CoppeliaSim y carga la escena").

Esto es importante para la UX del launcher: el usuario puede pulsar
"Launch" antes de arrancar CoppeliaSim, ver el error y arrancarlo sin
matar el proceso.

### 5.2 Resolución de handles

Al construir el bridge se llama `_resolve_handles(n_agvs)`, que recorre
los paths `/AGVs/AGV_i` y `/AGVs/AGV_i/AGV_i_Light` para cachear los
handles. Si falta alguno se lanza `RuntimeError` con un mensaje claro:
"ejecuta scene_builder.py primero". Esto evita errores crípticos cuando
el usuario olvida construir la escena.

### 5.3 Tres modos de renderizado

El sistema soporta tres backends seleccionables desde el launcher:

| Modo           | Render visible | Timing dirigido por |
|----------------|----------------|---------------------|
| `pygame`       | Solo Pygame (2D)| Pygame (`renderer.clock`) |
| `coppeliasim`  | Solo CoppeliaSim (3D) | `time.sleep(delay)` |
| `both`         | Pygame + CoppeliaSim | Pygame; CoppeliaSim interpola al mismo `t` |

#### Modo `coppeliasim` puro

El bucle llama `bridge.sync(env.agvs, interp_steps=8, step_delay=0.01)`.
Cada paso del entorno se subdivide en `interp_steps` sub-frames con
interpolación lineal de posición, produciendo un deslizamiento suave en
lugar de saltos celda a celda.

#### Modo `both` (Pygame + CoppeliaSim sincronizados)

El reto técnico aquí es que Pygame interpola el movimiento a 60 fps con
una animación propia controlada por `renderer._anim_t` (valor [0, 1]).
Si CoppeliaSim interpolara con `interp_steps` y `time.sleep` aparte,
ambos visualizadores se desincronizarían.

Solución: el bridge expone dos métodos para este modo:

- `sync_step(agvs, tasks)`: se llama **una vez por paso del entorno**.
  Calcula la posición destino, actualiza el cache `_step_start_pos`,
  rota el cuerpo del AGV y refresca los colores de estado y marcadores
  de tareas. **No mueve los AGVs todavía**.
- `sync_frame(agvs, anim_t)`: se llama **una vez por frame de Pygame**
  (~60 Hz). Recibe el diccionario `anim_t` que Pygame está usando para
  su propia animación y aplica `setObjectPosition(lerp(start, target, t))`
  a cada AGV. Como ambos visualizadores leen el mismo `t`, las
  animaciones están perfectamente sincronizadas.

### 5.4 Orientación del cuerpo según movimiento

`_update_heading(agv_id, from_pos, to_pos)` calcula
`atan2(dy, dx) + _HEADING_OFFSET` (un offset opcional por si el "frente"
visual del modelo no apunta a +X) y lo aplica al dummy raíz del AGV.
El modelo OmniPlatform y la luz de estado, como hijos del dummy, rotan
automáticamente. Si el AGV no se mueve (`|dx|, |dy| < 1e-6`) se
conserva el último heading.

### 5.5 Colores de luz de estado

Cada AGV tiene una pequeña esfera blanca sobre el modelo que codifica
su `AGVStatus`:

| Estado                | Color (RGB)        | Significado |
|-----------------------|--------------------|-------------|
| IDLE                  | blanco             | Sin tarea asignada |
| MOVING_TO_PICKUP      | amarillo           | Yendo a recoger |
| LOADING               | verde              | Cargando en la estación |
| MOVING_TO_DELIVERY    | naranja            | Yendo a entregar |
| UNLOADING             | azul               | Descargando |
| CHARGING              | cian               | Cargando batería |

Esto permite leer de un vistazo qué está haciendo cada AGV en la
vista 3D sin necesidad de mirar la consola.

### 5.6 Marcadores de tareas

`sync_tasks(tasks)` mantiene una pequeña esfera flotante para cada
tarea activa, con reglas:

- **PENDING / ASSIGNED**: esfera roja sobre la celda de pickup.
- **IN_PROGRESS**: esfera verde sobre la celda de delivery.
- **Transición pickup → delivery**: flash amarillo breve en la celda de
  pickup (visible un solo tick).
- **Tarea completada**: flash amarillo breve en la celda de delivery
  antes de desaparecer.

El bridge mantiene caches internos (`_task_markers`, `_task_status_cache`,
`_task_delivery`, `_flash_queue`) para detectar transiciones y limpiar
flashes en el siguiente tick. Esto da una pista visual clara de qué
tareas están en juego y cuándo se producen los eventos clave, sin
mirar la consola.

---

## 6. Integración con el launcher

El menú de lanzamiento (`src/rendering/launch_menu.py`) ofrece tres
botones de renderer: `pygame`, `coppeliasim`, `both`. Cuando el usuario
selecciona uno de los dos últimos:

1. Aparece un botón **"Build Scene"** que ejecuta `build_scene()` en un
   hilo de fondo. El usuario ve los estados `building → done / error`.
2. Al pulsar **"Launch"**, antes de arrancar, se lanza una comprobación
   de conexión (`_check_cs`) también en hilo de fondo, con timeout de
   5 s. Si tiene éxito, el lanzamiento procede; si falla, el botón
   LAUNCH muestra el error en rojo y permite reintentar.

Esto evita que la app se cuelgue si CoppeliaSim no está corriendo, y
da al usuario feedback claro de qué está pasando.

---

## 7. Mantenimiento de constantes compartidas

Tanto `scene_builder.py` como `bridge.py` necesitan compartir
constantes (tamaño de celda `0.5 m`, color ambiente `0`, etc.). Estas
están duplicadas en ambos archivos con un comentario `# Must match`
señalando el espejo. Es una duplicación consciente porque:

- Evita una dependencia cruzada (bridge no debe importar scene_builder).
- Son valores muy estables; no han cambiado desde el primer commit.

Si en el futuro se necesitan cambios frecuentes, lo natural sería
extraerlas a un módulo `src/coppeliasim/_constants.py`.

---

## 8. Limitaciones conocidas

- **Solo visualización**: la dinámica física de CoppeliaSim no se usa.
  Los AGVs no colisionan visualmente con muros (la rejilla del entorno
  ya garantiza que no atraviesan obstáculos, pero CoppeliaSim podría
  visualmente "atravesarlos" si hubiera un bug en `PlantMap`).
- **Un solo layout (L1)**: la escena se reconstruye desde `PlantMap`,
  que actualmente es estática. Soportar múltiples layouts implicaría
  parametrizar la rejilla, no la escena.
- **No streamea métricas a CoppeliaSim**: no hay overlay de "tareas
  completadas: N", "reward: X". Esto se ve en la consola o en Grafana.
- **Coste de visualización**: cada `sync_step` envía ~`n_agvs × 4`
  llamadas RPC. Para `n_agvs=8`, eso son ~32 llamadas por paso del
  entorno, lo que limita el modo `coppeliasim` a entornos de tamaño
  razonable (10–20 AGVs como mucho). No es problema para el TFM con
  ≤8 AGVs.
- **Sin paralelización**: como el bridge es estado por instancia, no
  se pueden correr varios bridges contra la misma instancia de
  CoppeliaSim. No es necesario para los experimentos planteados.

---

## 9. Decisiones descartadas y por qué

- **Escenas `.ttt` versionadas en git**: descartado por ser binarios
  opacos, difíciles de mantener entre versiones de CoppeliaSim, y
  romper la regla del repositorio de "todo el estado en código".
- **Plugin nativo / Lua scripts dentro de CoppeliaSim**: añadiría
  complejidad de desarrollo y debug (necesitaría reiniciar CoppeliaSim
  con cada cambio de Lua), sin aportar nada que la API ZMQ no cubra.
- **Bridge bidireccional (CoppeliaSim envía observaciones al entorno)**:
  rompería la reproducibilidad y haría imposible entrenar headless.
- **Simulación física activa**: descartada por los conflictos con los
  scripts internos de OmniPlatform; además, el entorno Gym ya implementa
  la dinámica relevante (movimiento celda a celda, colisiones evitadas
  por reservas de celda).
- **Cargar el modelo OmniPlatform como root directamente sin dummy
  intermedio**: descartado por dificultad de aplicar rotaciones Z
  limpias sobre la orientación natural del modelo.

---

## 10. Referencias rápidas

- Archivos principales:
  - `src/coppeliasim/scene_builder.py` — construcción de escena.
  - `src/coppeliasim/bridge.py` — sincronización runtime.
- Punto de entrada CLI directo:
  - `python -m src.coppeliasim.scene_builder --n_agvs 4`
- Integración con el launcher: botones "Build Scene" + selector de
  renderer en `src/rendering/launch_menu.py`.
- Dependencia: `pip install coppeliasim-zmqremoteapi-client`.
- Puerto por defecto: `localhost:23000`.
- Modelo usado: `Omnidirectional platform.ttm` (incluido en CoppeliaSim).
- Convención de ejes: Z-up, `(row, col)` → `(Y, X)`, 0.5 m por celda.
