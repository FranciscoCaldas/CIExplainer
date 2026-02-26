import os
import re
import ast
import random
import argparse
from collections import defaultdict
from itertools import combinations

import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration


# =====================================================
# CONSTANTS
# =====================================================

SYSTEM_PROMPT = (
    "You are an expert evaluator of GNN explanations. "
    "Answer precisely according to the instructions."
)

ANSWER_SUFFIX = "The better explanation is: "

SAMPLES_TO_USE = list(range(10))  # sample_ids to include: [0, 1, ..., 9]


# =====================================================
# PROMPT TEMPLATES
# =====================================================

PAIRWISE_PROMPT = """\
Task:
You are evaluating two GNN explanations. Given the graph structure, decide which explanation \
is overall better — considering accuracy, faithfulness to the structure, and clarity.

Rules:
- Only use the provided structural summary.
- Consider how well each explanation describes the explainable nodes and structural motifs.
- Consider how clearly and accurately the explanation communicates its reasoning.
- Answer only with A or B.

STRUCTURAL SUMMARY:
{structure}

EXPLANATION A:
{explanation_a}

EXPLANATION B:
{explanation_b}

Which explanation is overall better? Answer with just the letter A or B.\
"""


# =====================================================
# MODEL
# =====================================================

def load_model(model_name: str):
    """Load Gemma3 with 4-bit quantization and its processor."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


# =====================================================
# GRAPH UTILITIES
# =====================================================

def parse_graph_file(filepath: str):
    """
    Parse a GNN explainer output file.

    Returns:
        G             (nx.Graph)   : full graph
        prediction    (float)      : sigmoid prediction value
        explain_nodes (list[int])  : sorted top-5 explanation node indices
    """
    with open(filepath, "r") as f:
        content = f.read()

    pred_match = re.search(r"Prediction \(sigmoid\): ([0-9.]+)", content)
    prediction = float(pred_match.group(1)) if pred_match else None

    explain_match = re.search(r"Top-5 explanation nodes:\s*(\[[^\]]+\])", content)
    explain_nodes = sorted(ast.literal_eval(explain_match.group(1))) if explain_match else []

    edge_block_match = re.search(
        r"Number of edges:\s*\[\[(.*?)\]\]\s*Number of edges:",
        content,
        re.DOTALL,
    )
    if edge_block_match is None:
        raise ValueError(f"Could not parse edges in {filepath}")

    rows = edge_block_match.group(1).split("]\n [")
    row1 = list(map(int, rows[0].split()))
    row2 = list(map(int, rows[1].split()))

    G = nx.Graph()
    for u, v in zip(row1, row2):
        G.add_edge(u, v)

    return G, prediction, explain_nodes


def build_structure_summary(G: nx.Graph, explain_nodes: list) -> str:
    """
    Build a text summary of structural motifs (triangles, 4-cycles, 5-cycles)
    detected in the full graph, relative to the explanation nodes.
    """
    triangles = list(set(
        tuple(sorted(c)) for c in nx.enumerate_all_cliques(G) if len(c) == 3
    ))
    all_cycles = list(nx.simple_cycles(G.to_directed()))
    cycles_4 = list(set(tuple(sorted(c)) for c in all_cycles if len(c) == 4))
    cycles_5 = list(set(tuple(sorted(c)) for c in all_cycles if len(c) == 5))

    sections = [
        f"Explainable Nodes:\n{sorted(explain_nodes)}",
        "Triangles:\n"  + ("\n".join(map(str, triangles)) if triangles else "NONE"),
        "4-Cycles:\n"   + ("\n".join(map(str, cycles_4))  if cycles_4  else "NONE"),
        "5-Cycles:\n"   + ("\n".join(map(str, cycles_5))  if cycles_5  else "NONE"),
    ]
    return "\n\n".join(sections)


# =====================================================
# GENERATION & PARSING
# =====================================================

def generate_choice(
    prompt: str,
    suffix: str,
    model,
    processor,
    system_msg: str = SYSTEM_PROMPT,
) -> str:
    """
    Run a pairwise comparison prompt and return the raw decoded output.

    Args:
        prompt     : Evaluation prompt with placeholders filled.
        suffix     : Text appended after the chat template (e.g. "The better explanation is: ").
        model      : Loaded Gemma3 model.
        processor  : Corresponding AutoProcessor.
        system_msg : System message to prepend.

    Returns:
        Raw decoded string from the model.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {"role": "user",   "content": [{"type": "text", "text": prompt}]},
    ]

    formatted = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) + suffix

    inputs = processor(text=[formatted], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False)

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def extract_choice(text: str) -> str:
    """Parse 'A' or 'B' from model output; returns 'unclear' if ambiguous."""
    text = text.strip().upper().replace("**", "").replace("\n", "")

    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    if "EXPLANATION A" in text or " A " in text:
        return "A"
    if "EXPLANATION B" in text or " B " in text:
        return "B"
    return "unclear"


# =====================================================
# PAIR BUILDER
# =====================================================

def build_pairs(input_folders: list, samples_to_use: list) -> list:
    """
    Load explanation CSVs from each folder, assign a version label,
    filter to the requested sample IDs, then generate all cross-version
    pairwise combinations per graph file.

    Returns a list of pair dicts with keys:
        filename, graph_label, version_a, version_b, explanation_a, explanation_b
    """
    all_dfs = []
    for folder in input_folders:
        csv_path = os.path.join(folder, "all_llm_explanations_clean.csv")
        if not os.path.exists(csv_path):
            print(f"[SKIP] Missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["version"] = os.path.basename(os.path.dirname(folder))
        df = df[df["sample_id"].isin(samples_to_use)]
        all_dfs.append(df)

    by_file = defaultdict(list)
    for df in all_dfs:
        for _, row in df.iterrows():
            by_file[row["filename"]].append(row)

    pairs = []
    for filename, rows in by_file.items():
        if len(rows) < 2:
            continue
        for a, b in combinations(rows, 2):
            if random.random() < 0.5:
                a, b = b, a
            pairs.append({
                "filename":      filename,
                "graph_label":   a["graph_label"],
                "version_a":     a["version"],
                "version_b":     b["version"],
                "explanation_a": a["explanation"],
                "explanation_b": b["explanation"],
            })

    return pairs


# =====================================================
# REPORTING
# =====================================================

def print_summary(df: pd.DataFrame) -> None:
    """Print win-count and head-to-head summaries to stdout."""
    clear = df[df["winner"] != "unclear"]
    total = len(clear)

    print("\n=== OVERALL WIN COUNTS ===")
    print(df["winner"].value_counts().to_string())

    print("\n=== WIN RATE % ===")
    win_counts = clear["winner"].value_counts()
    print((win_counts / total * 100).round(2).astype(str).add("%").to_string())

    print("\n=== HEAD-TO-HEAD ===")
    for (va, vb), group in df.groupby(["version_a", "version_b"]):
        wins_a   = (group["winner"] == va).sum()
        wins_b   = (group["winner"] == vb).sum()
        unclear  = (group["winner"] == "unclear").sum()
        print(f"  {va} vs {vb}: {va}={wins_a} | {vb}={wins_b} | unclear={unclear}")

    print("\n=== WIN COUNTS BY GRAPH LABEL ===")
    print(
        clear.groupby(["graph_label", "winner"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )


# =====================================================
# MAIN
# =====================================================

def main(
    input_folders: list,
    graph_folder: str,
    model_name: str,
    output_path: str,
    samples_to_use: list,
    seed: int,
) -> None:

    random.seed(seed)

    model, processor = load_model(model_name)

    pairs = build_pairs(input_folders, samples_to_use)
    print(f"Total pairs to evaluate: {len(pairs)}")

    results = []
    for pair in tqdm(pairs, desc="Evaluating pairs"):
        graph_path = os.path.join(graph_folder, pair["filename"])
        G, _, explain_nodes = parse_graph_file(graph_path)
        structure_text      = build_structure_summary(G, explain_nodes)

        raw    = generate_choice(
            PAIRWISE_PROMPT.format(
                structure=structure_text,
                explanation_a=pair["explanation_a"],
                explanation_b=pair["explanation_b"],
            ),
            suffix=ANSWER_SUFFIX,
            model=model,
            processor=processor,
        )
        choice = extract_choice(raw)
        winner = pair["version_a"] if choice == "A" \
            else pair["version_b"] if choice == "B" \
            else "unclear"

        results.append({
            "filename":    pair["filename"],
            "graph_label": pair["graph_label"],
            "version_a":   pair["version_a"],
            "version_b":   pair["version_b"],
            "raw_answer":  raw,
            "winner":      winner,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")

    print_summary(results_df)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run pairwise LLM-as-judge evaluation on GNN explanation CSVs."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/models/gemma_3_27b",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="pairwise_judge_results_all.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for A/B position randomisation.",
    )
    args = parser.parse_args()

    GRAPH_FOLDER = "/llm_input_graph_sage"

    INPUT_FOLDERS = [
        "p4/llm_explanations_random",
        "p3/llm_explanations_random",
        "p2/llm_explanations_random",
        "p1/llm_explanations_random",
    ]

    main(
        input_folders=INPUT_FOLDERS,
        graph_folder=GRAPH_FOLDER,
        model_name=args.model_name,
        output_path=args.output_path,
        samples_to_use=SAMPLES_TO_USE,
        seed=args.seed,
    )