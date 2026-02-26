import os
import re
import ast
import random
import csv
import argparse

import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# =====================================================
# CONSTANTS & PROMPT TEMPLATES
# =====================================================

SYSTEM_PROMPT = (
    "You are an expert in Graph Neural Networks and graph structure analysis. "
    "Always respond with a direct analytical explanation in 4-6 sentences. "
    "Never repeat the instructions or write code."
)

CIRCLE_DEFINITION = "CIRCLE PATTERN:\nDefinition: A connected 5-node cycle."
HOUSE_DEFINITION  = "HOUSE PATTERN:\nDefinition: A 4-cycle with an additional triangle (roof)."
NONE_DEFINITION   = "NONE PATTERN:\nThe graph does not contain either the Circle or House structural motifs."

PROMPT_TEMPLATE = """\
You are analyzing why a Graph Neural Network (GNN) identified specific nodes as important for its prediction.

TASK:
You are NOT asked to explain why the graph is a "{label}".
You are asked to explain whether the EXPLAINABLE NODES support that prediction.

STRUCTURAL DEFINITIONS:
{circle}

{house}

{none}

GNN PREDICTION: "{label}"
EXPLAINABLE NODES: {explain_nodes}

STRUCTURES DETECTED IN SUBGRAPH:
{structure_text}

Write 4-6 analytical sentences.
Explanation:"""

CSV_HEADER = [
    "filename",
    "graph_label",
    "sample_id",
    "seed",
    "explanation",
]

GENERATION_KWARGS = dict(
    max_new_tokens=500,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
)


# =====================================================
# UTILITIES
# =====================================================

def set_all_seeds(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================
# PARSER
# =====================================================

def parse_llm_input_file(filepath: str) -> dict:
    """
    Parse a GNN explainer output file.

    Returns a dict with keys:
        - explain (torch.Tensor): top-5 explanation node indices
        - edges   (torch.Tensor): edge index tensor of shape [2, E]
        - label   (str)         : "House" | "Circle" | "Unknown"
    """
    with open(filepath, "r") as f:
        content = f.read()

    # ---- Label ----
    pred_match = re.search(r"Prediction \(sigmoid\):\s*([\d.]+)", content)
    if pred_match:
        label = "House" if float(pred_match.group(1)) >= 0.5 else "Circle"
    else:
        label = "Unknown"

    # ---- Explanation nodes ----
    explain_match = re.search(r"Top-5 explanation nodes:\s*(\[[^\]]+\])", content)
    if not explain_match:
        raise ValueError(f"Missing explanation nodes in {filepath}")
    explain = torch.tensor(ast.literal_eval(explain_match.group(1)))

    # ---- Edges ----
    edges_match = re.search(r"Number of edges:\s*(\[\[.*?\]\])", content, re.DOTALL)
    if not edges_match:
        raise ValueError(f"Missing edges block in {filepath}")

    raw = re.sub(r" +", " ", edges_match.group(1).replace("\n", " "))
    inner = re.findall(r"\[\s*([\d\s]+?)\s*\]", raw)
    if len(inner) < 2:
        raise ValueError(f"Could not parse edges in {filepath}")

    edges = torch.tensor([
        list(map(int, inner[0].split())),
        list(map(int, inner[1].split())),
    ])

    return {"explain": explain, "edges": edges, "label": label}


# =====================================================
# GRAPH UTILITIES
# =====================================================

def build_adjacency_text(edges_list: list) -> str:
    """Build a human-readable adjacency list string from an edge list."""
    adj = {}
    for u, v in edges_list:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    return "\n".join(
        f"node {node} -> {' '.join(map(str, sorted(neighbors)))}"
        for node, neighbors in sorted(adj.items())
    )


# =====================================================
# MODEL
# =====================================================

def load_model(model_name: str):
    """Load a 4-bit quantized causal LM and its tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return tokenizer, model


# =====================================================
# MAIN
# =====================================================

def main(
    input_folder: str,
    output_folder: str,
    num_samples: int,
    model_name: str,
    base_seed: int,
) -> None:

    set_all_seeds(base_seed)
    tokenizer, model = load_model(model_name)

    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "all_llm_explanations.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)

        for filename in tqdm(sorted(os.listdir(input_folder)), desc="Processing files"):
            if not filename.endswith(".txt"):
                continue

            input_path = os.path.join(input_folder, filename)
            try:
                parsed = parse_llm_input_file(input_path)
            except Exception as e:
                print(f"[ERROR] {filename}: {e}")
                continue

            explain_nodes  = sorted(parsed["explain"].tolist())
            edges          = parsed["edges"]
            label          = parsed["label"]

            edges_list     = list(zip(edges[0].tolist(), edges[1].tolist()))
            structure_text = build_adjacency_text(edges_list)

            prompt = PROMPT_TEMPLATE.format(
                label=label,
                circle=CIRCLE_DEFINITION,
                house=HOUSE_DEFINITION,
                none=NONE_DEFINITION,
                explain_nodes=explain_nodes,
                structure_text=structure_text,
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]

            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            print(f"\nProcessing {filename}")

            for sample_id in tqdm(range(num_samples), desc="  Sampling", leave=False):
                seed = base_seed + sample_id
                set_all_seeds(seed)

                with torch.no_grad():
                    output = model.generate(**inputs, **GENERATION_KWARGS)

                response    = tokenizer.decode(output[0], skip_special_tokens=True)
                explanation = response.split("Explanation:")[-1].strip() \
                    if "Explanation:" in response else response.strip()

                writer.writerow([
                    filename,
                    label,
                    sample_id,
                    seed,
                    explanation.replace("\n", " "),
                ])

    print(f"\n✅ All results saved to: {csv_path}")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM explanations on GNN explainer output files."
    )
    parser.add_argument("--input_folder",  type=str, default="llm_input_graph_sage")
    parser.add_argument("--output_folder", type=str, default="/p4/llm_explanations_random")
    parser.add_argument("--num_samples",   type=int, default=30)
    parser.add_argument("--model_name",    type=str, default="/models/Llama-3-8B-It")
    parser.add_argument("--base_seed",     type=int, default=1)
    args = parser.parse_args()

    main(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        num_samples=args.num_samples,
        model_name=args.model_name,
        base_seed=args.base_seed,
    )