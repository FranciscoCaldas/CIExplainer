import os
import re
import ast
import argparse

import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration


# =====================================================
# CONSTANTS
# =====================================================

GROUND_TRUTH_NODES = [20, 21, 22, 23, 24]

SYSTEM_PROMPT = (
    "You are an expert evaluator of GNN explanations. "
    "Answer precisely according to the instructions."
)

SCORE_SUFFIX = "Score: "


# =====================================================
# PROMPT TEMPLATES
# =====================================================

NODE_HALLUCINATION_PROMPT = """\
Task:
You are evaluating how well a GNN explanation handles its explainable nodes.

CONTEXT:
Ground-truth important nodes are always: 20, 21, 22, 23, 24.
Any explainable node NOT in this set is a noise node and must NOT be assigned structural importance.

EXPLAINABLE NODES PROVIDED TO THE EXPLANATION MODEL: {explainable_nodes}

Classify:
- SIGNAL NODES: nodes in {explainable_nodes} that are also in [20, 21, 22, 23, 24]
- NOISE NODES: nodes in {explainable_nodes} that are NOT in [20, 21, 22, 23, 24]

EXPLANATION TO EVALUATE:
{explanation}

STEP 1 — CRITICAL CHECK (apply before anything else):

The explanation is allowed to describe detected structures in full (e.g. "the 5-cycle is (20,21,22,23,24)")
even if some of those nodes are not in {explainable_nodes}. This is context, not a claim.

The ONLY thing that triggers a penalty is when the explanation explicitly claims that a noise node
(a node in {explainable_nodes} but NOT in [20,21,22,23,24]) IS PART OF, BELONGS TO, or
FORMS a detected structure.

Ask yourself: "Is the explanation saying this noise node has structural membership?"
- If YES → severe hallucination penalty (1 noise node → score 3 at most, 2+ → score 2)
- If NO (just describing the full structure or ignoring the noise node) → no penalty, proceed to STEP 2.

STEP 2 — SIGNAL NODES:
- Penalize if a signal node present in {explainable_nodes} is ignored, dismissed, or incorrectly described.
- Do NOT penalize for ground-truth nodes absent from {explainable_nodes} — the model can only discuss what it was given.

STEP 3 — NOISE NODES:
- Penalize only for active overclaiming (forcing structural roles onto noise nodes).
- Do NOT penalize for vagueness, uncertainty, or silence on noise nodes — these are correct behaviors.

Scoring:
- 5: All signal nodes correctly justified, no overclaiming on noise nodes.
- 4: Signal nodes mostly correct, at most minor overclaiming on noise nodes.
- 3: One signal node missed, OR noise nodes partially forced into the structure.
- 2: Multiple signal nodes missed, OR 1 noise node falsely assigned to a structure.
- 1: Signal nodes ignored entirely, OR 2+ noise nodes falsely assigned to structures.

Return your score and briefly justify it citing specific nodes.\
"""

STRUCTURE_ACCURACY_PROMPT = """\
Task:
You are checking whether a GNN explanation accurately describes the detected structures.

GROUND TRUTH STRUCTURES DETECTED:
{structure}

EXPLANATION TO EVALUATE:
{explanation}

Your job:
1. Check if every structure the explanation mentions actually exists in the ground truth.
2. Check if the explanation correctly identifies which explainable nodes belong to a detected structure.
3. Check if the explanation invents cycles or triangles not listed in the ground truth.
4. Check if the explanation incorrectly dismisses a node that IS part of a detected structure.

Scoring:
- 5: All structure references are accurate, no invented motifs, no missed memberships.
- 4: Minor inaccuracy in one node's membership, no invented structures.
- 3: One invented or misquoted structure, OR one node incorrectly dismissed or claimed.
- 2: Multiple structure errors or inventions.
- 1: Explanation structures are largely fabricated or wrong.

Return your score and cite the specific structure errors you found.\
"""

CLARITY_PROMPT = """\
Task:
You are evaluating the logical clarity of a GNN explanation. Your job is to determine how clearly
the explanation communicates the reasoning behind why the explainable nodes support the GNN's prediction.

Rules:
- Logical clarity includes coherence, step-by-step reasoning, and lack of ambiguity.
- Focus on whether a reader can easily understand the explanation without additional context.
- Partial credit is allowed if the explanation is somewhat clear but could be improved.

EXPLANATION TO EVALUATE:
{explanation}

Score from 1 (very poor) to 5 (perfect).\
"""


# =====================================================
# MODEL
# =====================================================

def load_model(model_name: str):
    """Load Gemma3 with 8-bit quantization and its processor."""
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
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
        G            (nx.Graph)  : full graph
        prediction   (float)     : sigmoid prediction value
        explain_nodes (list[int]): sorted top-5 explanation node indices
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

    rows  = edge_block_match.group(1).split("]\n [")
    row1  = list(map(int, rows[0].split()))
    row2  = list(map(int, rows[1].split()))

    G = nx.Graph()
    for u, v in zip(row1, row2):
        G.add_edge(u, v)

    return G, prediction, explain_nodes


def build_structure_summary(G: nx.Graph, explain_nodes: list) -> str:
    """
    Build a text summary of structural motifs (triangles, 4-cycles, 5-cycles)
    detected in the full graph, relative to the explanation nodes.
    """
    # ---- Triangles ----
    triangles = list(set(
        tuple(sorted(clique))
        for clique in nx.enumerate_all_cliques(G)
        if len(clique) == 3
    ))

    # ---- Cycles ----
    all_cycles = list(nx.simple_cycles(G.to_directed()))
    cycles_4 = list(set(tuple(sorted(c)) for c in all_cycles if len(c) == 4))
    cycles_5 = list(set(tuple(sorted(c)) for c in all_cycles if len(c) == 5))

    sections = [
        f"Explainable Nodes:\n{sorted(explain_nodes)}",
        "Triangles:\n" + ("\n".join(map(str, triangles)) if triangles else "NONE"),
        "4-Cycles:\n"  + ("\n".join(map(str, cycles_4))  if cycles_4  else "NONE"),
        "5-Cycles:\n"  + ("\n".join(map(str, cycles_5))  if cycles_5  else "NONE"),
    ]

    return "\n\n".join(sections)


# =====================================================
# GENERATION
# =====================================================

def generate_score(
    prompt: str,
    suffix: str,
    model,
    processor,
    system_msg: str = SYSTEM_PROMPT,
) -> str:
    """
    Run a single scoring prompt through the model and return the decoded output.

    Args:
        prompt     : Evaluation prompt with placeholders already filled.
        suffix     : Text appended after the chat template (e.g. "Score: ").
        model      : Loaded Gemma3 model.
        processor  : Corresponding AutoProcessor.
        system_msg : System message to prepend.

    Returns:
        Decoded string of newly generated tokens.
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

    inputs = processor(text=[formatted], return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=2,
            temperature=0.0,
            do_sample=False,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True)


# =====================================================
# MAIN
# =====================================================

def main(
    input_folders: list,
    graph_folders: list,
    model_name: str,
    output_filename: str,
) -> None:

    model, processor = load_model(model_name)

    for folder, graph_folder in zip(input_folders, graph_folders):
        input_csv = os.path.join(folder, "all_llm_explanations.csv")
        if not os.path.exists(input_csv):
            print(f"[SKIP] Missing: {input_csv}")
            continue

        df = pd.read_csv(input_csv)

        structural_scores = []
        hallucination_scores = []
        clarity_scores = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {folder}"):
            filename    = row["filename"]
            explanation = row["sub_nodes"]

            graph_path = os.path.join(graph_folder, filename)
            G, _, explain_nodes = parse_graph_file(graph_path)
            structure_text      = build_structure_summary(G, explain_nodes)

            structural_scores.append(generate_score(
                STRUCTURE_ACCURACY_PROMPT.format(
                    structure=structure_text,
                    explanation=explanation,
                ),
                suffix=SCORE_SUFFIX,
                model=model,
                processor=processor,
            ))

            hallucination_scores.append(generate_score(
                NODE_HALLUCINATION_PROMPT.format(
                    explainable_nodes=sorted(explain_nodes),
                    explanation=explanation,
                ),
                suffix=SCORE_SUFFIX,
                model=model,
                processor=processor,
            ))

            clarity_scores.append(generate_score(
                CLARITY_PROMPT.format(explanation=explanation),
                suffix=SCORE_SUFFIX,
                model=model,
                processor=processor,
            ))

        df["structural_score"]    = structural_scores
        df["hallucination_score"] = hallucination_scores
        df["clarity_score"]       = clarity_scores

        output_path = os.path.join(folder, output_filename)
        df.to_csv(output_path, index=False)
        print(f"✅ Saved to {output_path}")

    print("\nDone.")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge scoring on GNN explanation CSV files."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/data/rc.belo/models/gemma_3_27b",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="llm_judge_scores.csv",
    )
    args = parser.parse_args()

    INPUT_FOLDERS = [
        "p4/llm_explanations_random",
        "p3/llm_explanations_random",
        "p2/llm_explanations_random",
        "p1/llm_explanations_random",
        "p4_gin/llm_explanations_random",
        "p4_gcn/llm_explanations_random",
        "p4_gat/llm_explanations_random",
    ]

    GRAPH_FOLDERS = [
        "llm_input_graph_sage",
        "llm_input_graph_sage",
        "llm_input_graph_sage",
        "llm_input_graph_sage",
        "llm_input_gin",
        "llm_input_gcn",
        "llm_input_gat",
    ]

    main(
        input_folders=INPUT_FOLDERS,
        graph_folders=GRAPH_FOLDERS,
        model_name=args.model_name,
        output_filename=args.output_filename,
    )