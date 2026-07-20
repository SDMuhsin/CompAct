#!/bin/bash
# ============================================================================
# Pre-download all models and datasets to local cache
# ============================================================================
#
# Run this on a LOGIN NODE (with internet) before submitting sbatch jobs.
# Compute nodes have no internet access.
#
# Usage:
#   ./sbatch/download_cache.sh
#
# ============================================================================

set -e

source ./env/bin/activate

export HF_HOME=$(pwd)/data
export TORCH_HOME=$(pwd)/data
export HF_HUB_DISABLE_XET=1
mkdir -p $HF_HOME

echo "Cache directory: $HF_HOME"
echo ""

# ============================================================================
# Models
# ============================================================================

echo "=== Downloading models (files only, no loading) ==="

python -c "
from huggingface_hub import snapshot_download

models = [
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'huggyllama/llama-7b',
]

for model_name in models:
    print(f'Downloading {model_name}...')
    snapshot_download(repo_id=model_name)
    print(f'  Done: {model_name}')
"

echo ""

# ============================================================================
# Datasets
# ============================================================================

echo "=== Downloading datasets ==="

python -c "
import time
from datasets import load_dataset

# Large multi-shard downloads (esp. pg19's ~7GB / 23 shards) can hit transient
# network errors mid-stream (e.g. ChunkedEncodingError / IncompleteRead). With this
# script's 'set -e' a single hiccup would abort the WHOLE cache prep, so retry --
# load_dataset resumes from already-cached shards, so each attempt makes progress.
def _load(*args, _label=None, **kwargs):
    label = _label or '/'.join(str(a) for a in args)
    for attempt in range(1, 6):
        try:
            print(f'Downloading {label} (attempt {attempt})...', flush=True)
            ds = load_dataset(*args, **kwargs)
            print(f'  Done: {label}', flush=True)
            return ds
        except Exception as e:
            print(f'  Attempt {attempt} failed for {label}: {type(e).__name__}: {str(e)[:160]}', flush=True)
            if attempt == 5:
                raise
            time.sleep(10 * attempt)

# GLUE tasks
for task in ['cola', 'mrpc', 'sst2', 'rte', 'qnli', 'stsb']:
    _load('glue', task, _label=f'glue/{task}')

# SuperGLUE tasks
for task in ['boolq', 'cb']:
    _load('super_glue', task, _label=f'super_glue/{task}')

# WikiText-2
_load('wikitext', 'wikitext-2-raw-v1', _label='wikitext-2')

# Modern / challenging tasks (CONTEXT.md section 25). All are parquet/jsonl with NO
# loading script -- required since datasets>=4.0 dropped script support (so the old
# deepmind/pg19 and dynabench/dynasent scripts will NOT load).
_load('facebook/anli', _label='facebook/anli (anli_r1/r2/r3)')
_load('alisawuffles/WANLI', _label='alisawuffles/WANLI')
_load('tasksource/folio', _label='tasksource/folio')

# PG-19 long-context book LM (~7GB / 23 shards). Must be FULLY cached here: compute
# nodes run offline (HF_HUB_OFFLINE=1), so train-time streaming is impossible --
# train_glue.py loads the cached copy and subsets it (first N books, default 1000
# train / 50 eval). The full download_and_prepare must complete online so the offline
# load just reads prepared arrow.
_load('emozilla/pg19', _label='emozilla/pg19 (~7GB, this takes a while)')

# Commonsense reasoning multiple-choice suite (CONTEXT.md section 26 / LLM-Adapters).
# Train ONCE on Commonsense-170K, evaluate on all 8 sets. All parquet-backed (NO
# loading script -> load under datasets>=4.0). Used by src/commonsense_mc.py.
_load('zwhe99/commonsense_170k', _label='zwhe99/commonsense_170k (train, 170,420 rows)')
_load('google/boolq', _label='google/boolq (eval)')
_load('nthngdy/piqa', _label='nthngdy/piqa (eval; ybisk/piqa ships a script -> use this mirror)')
_load('lighteval/siqa', _label='lighteval/siqa (eval; social_i_qa ships a script -> use this mirror)')
_load('Rowan/hellaswag', _label='Rowan/hellaswag (eval)')
_load('allenai/winogrande', 'winogrande_xl', _label='allenai/winogrande winogrande_xl (eval)')
_load('allenai/ai2_arc', 'ARC-Easy', _label='allenai/ai2_arc ARC-Easy (eval)')
_load('allenai/ai2_arc', 'ARC-Challenge', _label='allenai/ai2_arc ARC-Challenge (eval)')
_load('allenai/openbookqa', 'main', _label='allenai/openbookqa main (eval)')
"

echo ""

# ============================================================================
# Evaluation metrics
# ============================================================================

echo "=== Downloading evaluation metrics ==="

python -c "
import evaluate

metrics = ['glue', 'super_glue', 'accuracy', 'f1', 'matthews_correlation', 'pearsonr', 'spearmanr', 'perplexity']
for m in metrics:
    try:
        print(f'Downloading metric: {m}...')
        evaluate.load(m)
        print(f'  Done: {m}')
    except Exception as e:
        print(f'  Skipped {m}: {e}')
"

echo ""
echo "============================================"
echo "All downloads complete."
echo "Cache directory: $HF_HOME"
echo "============================================"
