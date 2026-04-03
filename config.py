# ── config.py ─────────────────────────────────────────────────────────────────
# Single source of truth for all tunable constants.
# Import this everywhere instead of hardcoding values.

# ── Track ─────────────────────────────────────────────────────────────────────
TRACK_FILE      = "data/tracks/f1.csv"   # path to TUMFTM CSV
TRACK_SCALE     = 1.0                        # metres per unit (TUMFTM is already in m)

# ── Car physics ───────────────────────────────────────────────────────────────
WHEELBASE       = 2.5        # metres  (F1 ~3.6 m; keep smaller for stability)
MAX_STEER       = 0.8        # radians (~28°)
MAX_ACCEL       = 15.0       # m/s² (massively bumped to allow snappy acceleration)
MAX_BRAKE       = 15.0       # m/s²
MAX_SPEED       = 60.0       # m/s  (~288 km/h)
FRICTION        = 5.5        # m/s² (baseline rolling resistance / engine braking)
DRAG            = 0.001      # aerodynamic drag coefficient (scales quadratically with speed)
CORNERING_G     = 2.0        # max lateral G-force (limits turning radius at high speeds)
TURN_FRICTION   = 10.0        # multiplier for speed scrubbing during turns
DT              = 0.05       # seconds per step (20 Hz)

# ── Sensors ───────────────────────────────────────────────────────────────────
N_SENSORS       = 7
SENSOR_ANGLES   = [-60, -40, -20, 0, 20, 40, 60]  # degrees
SENSOR_MAX_DIST = 50.0       # metres — normalisation denominator

# ── Reward ────────────────────────────────────────────────────────────────────
REWARD_MODE           = "checkpoint"   # "checkpoint" | "laptime"
CHECKPOINT_REWARD     = 1.0
OFF_TRACK_PENALTY     = -10.0
SPEED_WEIGHT          = 0.01           # used in laptime mode
STEP_PENALTY          = 0.01           # penalty per step to encourage movement

# ── Training ──────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 5
LEARNING_RATE   = 3e-4
BATCH_SIZE      = 256
BUFFER_SIZE     = 1_000_000
N_ENVS          = 1          # override with --n-envs on cluster

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR       = "models/"
LOG_DIR         = "logs/"
TRAJECTORY_DIR  = "outputs/trajectories/"
PLOT_DIR        = "outputs/plots/"

# ── Rendering ─────────────────────────────────────────────────────────────────
WINDOW_W        = 1000
WINDOW_H        = 700
FPS             = 60
RENDER_CAMERA_MODE = "full"   # "follow" | "full"
HEADLESS        = False      # set True on cluster via --headless flag
