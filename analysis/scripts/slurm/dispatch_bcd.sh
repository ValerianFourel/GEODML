#!/usr/bin/env bash
# Dispatch Stages B → C+D (and optionally F) for the 4 active variants on
# JUPITER Booster via sbatch with explicit dependencies.
#
# Pipeline shape:
#
#   Stage B (features.py)   4 parallel jobs           ── per (engine × pool)
#                                  │
#                                  ▼  --dependency=afterok on ALL 4 features
#   Stage C+D (run_dml.sbatch)     4 jobs             ── one per variant
#                                  │                     (Stage C inline at job start)
#                                  ▼  --dependency=afterok on per-variant DML
#   Stage F (--with-stage-f)       N jobs             ── ablation/saliency/probing/weights
#                                                        per (variant × model × method)
#
# Reuses scripts/slurm/run_features.sbatch, run_dml.sbatch, run_ablation.sbatch,
# run_saliency.sbatch, run_probing.sbatch, run_weights.sbatch — each of which
# already sources _common.sh (sets HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1,
# activates venv, loads modules in the right order, and chain_resubmits on
# walltime expiry).
#
# Required env (or .env):
#   JUWELS_ACCOUNT       — SLURM accounting (e.g. "scifi")
#   GEODML_DATA_ROOT     — dataset root
#   JUWELS_PROJECT       — (optional) jutil project tag
#
# Usage:
#   ./scripts/slurm/dispatch_bcd.sh                       # B + C+D for 4 variants
#   ./scripts/slurm/dispatch_bcd.sh --with-stage-f         # also queue Stage F (~80 jobs)
#   ./scripts/slurm/dispatch_bcd.sh --dry-run              # print sbatch invocations only
#   ./scripts/slurm/dispatch_bcd.sh --variants biased,neutral
#
# Idempotent: features.py and dml.py both pass --resume, so re-running picks
# up partial state. Stage F jobs also self-skip via their .done markers.

set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

[ -f .env ] && { set -a; source .env; set +a; }
: "${JUWELS_ACCOUNT:?JUWELS_ACCOUNT must be set in .env or shell}"

# ── Defaults ─────────────────────────────────────────────────────────────────
VARIANTS=(biased neutral biased_rag neutral_rag)
ENGINES=(searxng ddg)
POOLS=(20 50)
MODELS=("meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen2.5-72B-Instruct")
TREATMENTS=(T7_source_earned T5_topical_comp T3_structured_data_new T2a_question_headings T6_freshness T1b_stats_density)
FRAMES=(full robust_winners)

WITH_STAGE_F=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --with-stage-f) WITH_STAGE_F=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --variants)     IFS=',' read -r -a VARIANTS <<< "$2"; shift 2 ;;
    --models)       IFS=',' read -r -a MODELS   <<< "$2"; shift 2 ;;
    --engines)      IFS=',' read -r -a ENGINES  <<< "$2"; shift 2 ;;
    --pools)        IFS=',' read -r -a POOLS    <<< "$2"; shift 2 ;;
    --help|-h)      sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p logs

EXPORTS_BASE="ATTEMPT=1,MAX_ATTEMPTS=6"
[ -n "${JUWELS_PROJECT:-}" ] && EXPORTS_BASE="$EXPORTS_BASE,JUWELS_PROJECT=$JUWELS_PROJECT"

# submit jobname depend script [export key=val ...]
# Returns the JID on stdout, status messages on stderr.
submit() {
  local jobname="$1"; shift
  local depend="$1"; shift
  local script="$1"; shift

  local exports="$EXPORTS_BASE"
  for kv in "$@"; do exports="$exports,$kv"; done

  local cmd=(
    sbatch
    --account="$JUWELS_ACCOUNT"
    --job-name="$jobname"
    --export="ALL,$exports"
  )
  [ -n "$depend" ] && cmd+=(--dependency="afterok:$depend")
  cmd+=("$script")

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry] %s\n' "${cmd[*]}" >&2
    # Emit a fake numeric JID so dependency chains in the dry-run look real.
    echo $((RANDOM % 900000 + 100000))
    return
  fi

  local out
  out=$("${cmd[@]}")
  printf '  %-40s -> %s\n' "$jobname" "$out" >&2
  echo "$out" | awk '/Submitted batch job/ {print $4}'
}

# ── Stage B: features (4 parallel, no deps) ──────────────────────────────────
echo "[Stage B] features — $((${#ENGINES[@]} * ${#POOLS[@]})) parallel jobs"
FEAT_JIDS=()
for E in "${ENGINES[@]}"; do
  for P in "${POOLS[@]}"; do
    jid=$(submit "feat-${E}-${P}" "" \
      scripts/slurm/run_features.sbatch \
      "ENGINE=$E" "POOL=$P" "FEATURES_DEVICE=cuda")
    [ -n "$jid" ] && FEAT_JIDS+=("$jid")
  done
done
FEAT_DEP=$(IFS=:; echo "${FEAT_JIDS[*]:-}")
echo

# ── Stages C+D: DML per variant (each waits on ALL features) ─────────────────
# Parallel arrays for variant -> JID lookup (portable across bash versions).
echo "[Stage C+D] DML — ${#VARIANTS[@]} jobs (each --dependency=afterok:$FEAT_DEP)"
DML_VAR_KEYS=()
DML_VAR_JIDS=()
for V in "${VARIANTS[@]}"; do
  jid=$(submit "dml-${V}" "$FEAT_DEP" \
    scripts/slurm/run_dml.sbatch \
    "PROMPT_VARIANT=$V")
  if [ -n "$jid" ]; then
    DML_VAR_KEYS+=("$V")
    DML_VAR_JIDS+=("$jid")
  fi
done
echo

dml_jid_for() {
  local target="$1"
  local i
  for i in "${!DML_VAR_KEYS[@]}"; do
    if [ "${DML_VAR_KEYS[$i]}" = "$target" ]; then
      echo "${DML_VAR_JIDS[$i]}"
      return
    fi
  done
  echo ""
}

# ── Stage F (optional): per (variant × model × method) ───────────────────────
STAGE_F_TOTAL=0
if [ "$WITH_STAGE_F" -eq 1 ]; then
  echo "[Stage F] interp — per (variant × model × method), --dependency on matching DML"
  for V in "${VARIANTS[@]}"; do
    DEP=$(dml_jid_for "$V")
    if [ -z "$DEP" ] && [ "$DRY_RUN" -ne 1 ]; then
      echo "  WARN: no DML JID for variant=$V; submitting Stage F without dep" >&2
    fi
    for M in "${MODELS[@]}"; do
      TAG="${M##*/}"
      # ablation × 6 treatments
      for T in "${TREATMENTS[@]}"; do
        submit "abl-${TAG}-${T}-${V}" "$DEP" \
          scripts/slurm/run_ablation.sbatch \
          "MODEL=$M" "TREATMENT=$T" "PROMPT_VARIANT=$V" > /dev/null
        STAGE_F_TOTAL=$((STAGE_F_TOTAL + 1))
      done
      # saliency × 2 frames
      for F in "${FRAMES[@]}"; do
        submit "sal-${TAG}-${F}-${V}" "$DEP" \
          scripts/slurm/run_saliency.sbatch \
          "MODEL=$M" "FRAME=$F" "PROMPT_VARIANT=$V" > /dev/null
        STAGE_F_TOTAL=$((STAGE_F_TOTAL + 1))
      done
      # probing (frame=both, single job)
      submit "prob-${TAG}-${V}" "$DEP" \
        scripts/slurm/run_probing.sbatch \
        "MODEL=$M" "PROMPT_VARIANT=$V" > /dev/null
      STAGE_F_TOTAL=$((STAGE_F_TOTAL + 1))
      # weights (if the sbatch exists)
      if [ -f scripts/slurm/run_weights.sbatch ]; then
        submit "wgt-${TAG}-${V}" "$DEP" \
          scripts/slurm/run_weights.sbatch \
          "MODEL=$M" "PROMPT_VARIANT=$V" > /dev/null
        STAGE_F_TOTAL=$((STAGE_F_TOTAL + 1))
      fi
    done
  done
  echo
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo "=== Submitted ==="
echo "  Stage B (features): ${#FEAT_JIDS[@]} jobs"
echo "  Stage C+D (DML):    ${#DML_VAR_KEYS[@]} jobs"
[ "$WITH_STAGE_F" -eq 1 ] && echo "  Stage F (interp):   $STAGE_F_TOTAL jobs"
echo
if [ "$DRY_RUN" -ne 1 ]; then
  echo "Monitor:"
  echo "  squeue -u \$USER --format='%.10i %.40j %.2t %.10M %.10L' | head -50"
  echo "  watch -n 60 'squeue -u \$USER | wc -l; .venv/bin/python scripts/audit_pipeline.py | tail -20'"
  echo
  echo "After queue drains, headline check:"
  echo "  python -c \"import pandas as pd, os; root=os.environ['GEODML_DATA_ROOT']; "
  echo "    [print(v, pd.read_parquet(f'{root}/data/dml_results/dml_results_long_{v}.parquet').query("
  echo "    \\\"subset=='POOLED' and method=='plr' and learner=='lgbm' and outcome=='rank_delta' and treatment=='T7_source_earned'\\\").iloc[0][['coef','p_val','n_obs']].to_dict()) for v in ['biased','neutral','biased_rag','neutral_rag']]\""
fi
