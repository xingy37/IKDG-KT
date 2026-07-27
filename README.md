# IKDG-KT

Official implementation of **IKDG-KT**, the Interaction-Knowledge Dual-Level Dynamic Graph framework for knowledge tracing. The model represents learning records as a continuous-time heterogeneous graph over students, questions, and knowledge concepts (KCs), and predicts whether a student will answer a question correctly.

The implementation accompanies the paper [Interaction-Knowledge Dual-Level Dynamic Graph for Knowledge Tracing](Interaction-Knowledge%20Dual-Level%20Dynamic%20Graph.pdf).

## Overview

Knowledge tracing requires both personalized response history and transferable concept-level evidence. IKDG-KT combines these two sources in one causal, dynamic graph model:

- **Interaction Layer - Event-Level Dynamic Encoding:** samples temporally valid student and question neighborhoods, composes node, edge, time, and structural features, and encodes the two histories with separate GRU updates.
- **Knowledge Layer - Time-Aware Concept Evolution:** maintains one state per KC, retrieves the concept-conditioned response history through QKV attention, combines current-question and historical evidence with an adaptive gate, and updates the KC state with a GRU.
- **Joint objective:** applies binary cross-entropy only to student-question response edges and a margin-based auxiliary objective to preserve question-KC structural alignment.

For a response event `(student, question, KC, result, time)`, IKDG-KT produces a prediction from the concatenated student, question, and evolving KC representations. The current response is appended to the KC history only after prediction, preventing target-label leakage.

## Repository Layout

```text
models/IKDG_KT.py                 IKDG-KT backbone
models/kc_history_manager.py      Per-KC chronological history store
train_link_classification.py      Training entry point
evaluate_link_classification.py   Checkpoint evaluation entry point
evaluate_models_utils.py          IKDG-KT-aware evaluation routine
processed_data/dbe_kt22_higher/  Included processed DBE-KT22 data
```

`IKDG-KT` is the command-line model name. `IKDG_KT` is the Python class and module spelling required by Python identifiers.

## Data

This release includes the processed `dbe_kt22_higher` split under `processed_data/`. It contains the temporal edge table, node and edge features, and `mapping_info.pkl`, which records the student, question, and KC node mappings required by IKDG-KT.

The included mapping defines 1,264 student nodes, 212 question nodes, and 98 KC nodes. To use another dataset, place files following the existing convention in `processed_data/<dataset_name>/`:

```text
ml_<dataset_name>.csv
ml_<dataset_name>.npy
ml_<dataset_name>_node.npy
mapping_info.pkl
```

`mapping_info.pkl` must provide `student_to_node_id`, `question_to_node_id`, and `kc_to_node_id` mappings for concept-aware training.

## Environment Setup

Python 3.9 is recommended. The project was developed with PyTorch 1.8.1; newer compatible PyTorch releases are supported. All Python dependencies are listed in [requirements.txt](requirements.txt).

### Conda

```bash
conda create -n ikdg-kt python=3.9 -y
conda activate ikdg-kt
pip install -r requirements.txt
```

### venv

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux or macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### CUDA-enabled PyTorch

For GPU training, install the PyTorch wheel that matches the local CUDA runtime from the [PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining dependencies. For example, for CUDA 11.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Confirm that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Training

The default configuration is the included `dbe_kt22_higher` dataset and `IKDG-KT` model. A representative three-run experiment is:

```bash
python train_link_classification.py --dataset_name dbe_kt22_higher --model_name IKDG-KT --num_neighbors 100 --batch_size 2000 --use_time_decay_in_attention --num_runs 3 --gpu 0
```

Important options:

| Option | Description | Default |
| --- | --- | --- |
| `--num_neighbors` | Historical neighbors sampled for each student and question | `50` |
| `--batch_size` | Chronological event batch size | `2000` |
| `--num_epochs` | Maximum number of epochs | `100` |
| `--learning_rate` | Adam learning rate | `0.0005` |
| `--lambda_struct` | Structural-loss coefficient in the IKDG-KT implementation | `0.1` |
| `--lambda_decay` | Temporal decay coefficient for concept-history retrieval | `0.1` |
| `--use_time_decay_in_attention` | Enables exponential time decay in history attention | disabled |
| `--max_history_length` | Retained interactions per KC memory bank | `10` |
| `--num_runs` | Number of random-seed runs | `3` |

Checkpoints are written to `saved_models/IKDG-KT/<dataset_name>/`, logs to `logs/IKDG-KT/`, and per-run metrics to `saved_results/IKDG-KT/`.

## Evaluation

Evaluate saved checkpoints with the same model name and dataset:

```bash
python evaluate_link_classification.py --dataset_name dbe_kt22_higher --model_name IKDG-KT --num_neighbors 100 --batch_size 2000 --use_time_decay_in_attention --num_runs 3 --gpu 0
```

The evaluator reports standard link-classification metrics for both regular and new-node splits. IKDG-KT computes prediction metrics only on student-question response edges; question-KC edges are used for structural supervision rather than response labels.

## Reproducibility Notes

- Events are consumed in chronological order. Do not shuffle the temporal edge stream.
- The KC history manager is stateful within a run. It stores only history preceding the current interaction and retains a bounded recent window.
- The model initializes node-type counts from `processed_data/<dataset_name>/mapping_info.pkl`. Missing mappings disable explicit KC-aware behavior, so provide the mapping file for all IKDG-KT experiments.
- Output directories are named with the public model identifier `IKDG-KT`.



