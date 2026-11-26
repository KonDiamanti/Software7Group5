#!/bin/bash
#SBATCH --job-name=sw7_tests
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# runs the tests of scripts/scale_and_reports.py on the AI-LAB.
# every test writes results/<name>.txt and results/<name>.done when it is finished,
# so if the job is submitted again it skips the tests that are finished.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p results logs
export PYTHONUNBUFFERED=1
export HF_TOKEN="$(cat "$HOME/.hf_token")"   # for Prompt Guard 2, made by the read -s step in the instructions

SIF="$(ls -t /ceph/container/pytorch/*.sif | head -1)"
echo "container: $SIF"
date

singularity exec --nv "$SIF" bash -c '
set -u
pip install --user -q fastapi httpx pydantic sentence-transformers datasets transformers scikit-learn scipy joblib langdetect pandas 2>&1 | tail -3

echo "--- check: unit tests of the new script"
python -m unittest tests.test_scale_and_reports 2>&1 | tail -4

echo "--- check: tier 1 model loads"
if ! python -c "from src.tier1.char_svm import CharSvmTier1; CharSvmTier1.load(\"models/tier1_char_svm_more_data.joblib\")"; then
  echo "the saved tier 1 model does not load here, training it again (about a minute)"
  python -m src.tier1.train --more-data
fi

run() {   # run NAME COMMAND...   (skips finished tests, goes on if one fails)
  name="$1"; shift
  if [ -f "results/$name.done" ]; then echo "--- $name: already done"; return; fi
  echo "--- $name: $*"
  { echo "$*"; "$@"; } 2>&1 | tee "results/$name.txt"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then touch "results/$name.done"; else echo "!!! $name failed, going on with the next test"; fi
}

# cheap ones first
run entry_test        python -m scripts.scale_and_reports entry
run scale_stub_70     python -m scripts.scale_and_reports scale --apps 10 --repeat 0.7 --runs 20 --csv results/scale_stub_70.csv
run scale_stub_30     python -m scripts.scale_and_reports scale --apps 10 --repeat 0.3 --runs 20 --csv results/scale_stub_30.csv
# with Prompt Guard 2 on the GPU
run reports_pg2_70    python -m scripts.scale_and_reports reports --guard prompt-guard-2 --device cuda --apps 10 --repeat 0.7 --runs 5 --csv results/reports_pg2_70.csv
run scale_pg2_70      python -m scripts.scale_and_reports scale --guard prompt-guard-2 --device cuda --apps 10 --repeat 0.7 --runs 20 --extra-pool 500 --csv results/scale_pg2_70.csv
run scale_pg2_30      python -m scripts.scale_and_reports scale --guard prompt-guard-2 --device cuda --apps 10 --repeat 0.3 --runs 20 --extra-pool 500 --csv results/scale_pg2_30.csv
echo "--- all finished"
'
date
