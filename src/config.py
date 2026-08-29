"""Apatheia configuration: model, layer, paths, experiment constants.

The policy model and the emotion-measurement instrument are inherited from
ICMI-022 (through-the-valley): Qwen 3.5 27B at the pinned HF revision, with
the 171-emotion direction basis extracted at decoder layer 53. The revision
pin is what keeps the bundled vectors valid — do not bump it casually.

"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"
TRAIN_DIR = DATA_DIR / "train"
PROMPTS_DIR = DATA_DIR / "prompts"
VECTORS_DIR = PROJECT_ROOT / "vectors"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
JUDGE_CACHE_DIR = DATA_DIR / "judge_cache"

# ── Policy model (ICMI-022 pin) ───────────────────────────────────────
MODEL_ID = "Qwen/Qwen3.5-27B"
MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"  # last_modified 2026-04-24
HIDDEN_DIM = 5120
NUM_LAYERS = 64
LAYER = 53          # readout layer of the 171-emotion basis
POOL_SKIP = 4       # skip the BOS region when mean-pooling prompts (ICMI-022 convention)

EMOTION_VECTORS_PATH = VECTORS_DIR / "emotion_vectors_qwen27b_best.pt"
EMOTION_META_PATH = VECTORS_DIR / "meta_qwen27b.json"
HEADLINE_EMOTIONS = ["afraid", "sad", "happy", "calm"]  # the Anthropic-figure four

# ── Generation (policy replies; thinking disabled — the reply IS the object) ──
# Amendment 2: 256 truncated 87% of BASE replies mid-sentence (the model's
# natural length on these prompts is ~200+ words), polluting the helpfulness
# channel. The truncation gate (<10%) now enforces cap adequacy: 512 still
# truncated 26.2% of base eval replies (gate caught it, 2026-08-22 ~21:00),
# so 1024 it is.
MAX_NEW_TOKENS = 1024
GENERATE_TIMEOUT_SECONDS = 480.0
TRUNCATION_GATE = 0.10
EVAL_N_SAMPLES = 20        # per eval cell, sampled (ICMI-022: even n, temp 0.7)
EVAL_TEMPERATURE = 0.7

# ── Judge ─────────────────────────────────────────────────────────────
JUDGE_MODEL = "claude-opus-5"
JUDGE_CONCURRENCY = 30
JUDGE_MAX_RETRIES = 5

# Reward: unemotional AND still competent.
#   r = (5 - emotionality)/5 * min(helpfulness, HELPFUL_FULL)/HELPFUL_FULL
# The helpfulness factor blocks the degenerate optimum (empty/refusal replies
# are perfectly unemotional). Soft gate, full credit at HELPFUL_FULL.
HELPFUL_FULL = 3

# ── Data generation (one-time, laptop) ────────────────────────────────
GEN_MODEL = "claude-opus-5"
N_PARAPHRASES = 10          # canonical + 9 rewrites per eval template
TRAIN_PER_CATEGORY = 20     # 12 categories x 20 = 240 training prompts

SEED = 42
