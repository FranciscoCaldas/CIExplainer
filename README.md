#  CIExplainer

> **CIExplainer: Generating Causal and Interpretable Explanations for Graph Neural Networks**

CIExplainer is the official implementation accompanying the paper of the same name. It provides a framework for generating human-readable, causally-grounded explanations for predictions made by Graph Neural Networks (GNNs), combining subgraph-based structural explanations with large language model (LLM) narration.

paper available at:
https://arxiv.org/abs/2606.20747


---
The framework supports both **node classification** and **graph classification** tasks. And provides Explanations for 5 different explanation Methods:
RandomExplainer, GNNExplainer, PGExplainer, SubgraphX and CIExplainer (Ours).
## Datasets, Models, Explainers

### Supported datasets
- NC: `ba_shapes`, `tree_grid`, 
- GC: `ba_2motif`, `mutag`

### Supported models
- `gcn`, `graphsage`, `gat`, `gin`

### Supported explainers
- `random_explainer`
- `gnnexplainer`
- `pgexplainer`
- `subgraphx`
- `ciexplainer`

---

## Usage

### 1. Train a GNN

```bash
python gnn_train.py --task gc --model all --dataset all --num_epochs 1000 --lr 0.001 --batch_size 32
```

Arguments:
- `--task`: `nc`, `gc`, `lp`, `all`
- `--model`: model name or `all`
- `--dataset`: dataset name or `all`
- `--num_epochs`: number of epochs
- `--lr`: learning rate
- `--batch_size`: batch size

Models are saved to the `models/` directory.

## Explainability Evaluation

### Python directly

Recommended script:

```bash
python gnn_explain_std.py --task all --model all --dataset all --explainer all --num_runs 10
```

Arguments:
- `--task`: `nc`, `gc`, `lp`, `all`
- `--model`: `gcn`, `gin`, `gat`, `graphsage`, `all`
- `--dataset`: task-appropriate dataset or `all`
- `--explainer`: `random`, `gnnexplainer`, `pgexplainer`, `subgraphx`, `ciexplainer`, `all`
- `--num_runs`: repetition count

### 3. Generate LLM Explanations

```bash
python generate_llm_explanations_p4.py --input_folder --output_folder --numsamples --model_name "/models/Llama-3-8B-It"
```

## Repository Structure

```
CIExplainer/
│
├── ciexplainer.py               # Core CIExplainer logic
│                                ##training
├── gnn_train.py                 # GNN training entry point
├── gnn_lightning.py             # PyTorch Lightning GNN module
├── graph_classification.py      # Graph classification task utilities
├── node_classification.py       # Node classification task utilities

│                                #Explanaition
├── gnn_explain_std.py           # GNN_explanation entry point
│
├── explain_gc.py                # Explanation pipeline for graph classification
├── explain_nc.py                # Explanation pipeline for node classification
│
│                                #LLM description
├── generate_llm_explanations_p4.py  # LLM-based explanation generation
├── prompts.py                   # Prompt templates for LLM calls
│
├── llm_as_judge_eval.py         # LLM-as-judge single-score evaluation
├── llm_as_judge_pairwise.py     # LLM-as-judge pairwise comparison evaluation
│
│                                #Utils
├── model_store.py               # Model loading/saving utilities
├── data_store.py                # Dataset loading/saving utilities
├── explainer_store.py           # Explainer configuration store
│
├── pre_process_MUTAG.py         # MUTAG dataset preprocessing
├── utils.py                     # General-purpose utilities
├── subgraphX_explainer.py       # SubgraphX explainer wrapper
├── subgraphX_mcts.py            # Monte Carlo Tree Search for SubgraphX
├── subgraphX_utils.py           # SubgraphX helper utilities
├── datasets/                    # Dataset files
├── models/                      # Pretrained/saved GNN models
├── figures_example/             # Example output figures
└── llm_input/                   # Preprocessed inputs for the LLM stage
```

---
