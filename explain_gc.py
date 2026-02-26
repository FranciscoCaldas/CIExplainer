import time
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import os
from datetime import datetime

from torch_geometric.explain import ModelConfig
from torch_geometric.explain.metric import groundtruth_metrics, fidelity, characterization_score, unfaithfulness
from torchmetrics.functional.classification import binary_jaccard_index

from utils import setup_models, custom_iou, custom_precision, custom_recall, custom_fidelity
from model_store import get_gnn
from explainer_store import get_explainer


def count_triangles_per_node(edge_index, num_nodes):
    """Count the number of triangles each node participates in."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.cpu().t().numpy()
    G.add_edges_from(edges)
    
    triangles = nx.triangles(G)
    return triangles

def visualize_graph_explanation_single(edge_index, gt_edge_mask,gt_node_mask, explain_node_mask,
                                       dataset_name, model_name, explainer_name,
                                       graph_idx, save_dir='figures_viz'):

    os.makedirs(save_dir, exist_ok=True)

    # Build graph
    G = nx.Graph()
    edges = edge_index.cpu().t().numpy()
    num_nodes = int(edge_index.max()) + 1

    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    #pos = nx.planar_layout(G) if nx.check_planarity(G)[0] else pos
    fig, ax = plt.subplots(figsize=(7, 7))



    motif_nodes = [i for i in range(len(gt_node_mask)) if gt_node_mask[i] == 1]
    
    # ---- Create circular layout for motif ----
    circle_pos = nx.circular_layout(motif_nodes, scale=0.5)

    # ---- Overwrite their positions in full layout ----

    for n in motif_nodes[1:]:
        pos[n] = circle_pos[n] + pos[motif_nodes[0]]  

    #for n in pos:
    #    if n not in motif_nodes:
    #        pos[n] *= 1.4
    # ---- Base graph (light) ----
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        edge_color='gray',
        width=1.2,
        alpha=0.9
    )

    # ---- Ground truth motif edges (RED) ----
    gt_edge_mask_t = gt_edge_mask.detach().cpu().numpy().T
    #gt_edges = [tuple(edges[i]) for i in range(len(edges)) if gt_edge_mask[i] == 1]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=gt_edge_mask_t,
        ax=ax,
        edge_color='black',
        width=3.0,
        alpha=0.95
    )

    # ---- Explainer node coloring ----
    explain_mask = explain_node_mask.detach().cpu().numpy()
    max_val = explain_mask.max() if explain_mask.max() > 0 else 1

    node_colors = [
        cm.Blues(explain_mask[i] / max_val) if explain_mask[i] > 0
        else (0.9, 0.9, 0.9, 1.0)
        for i in range(num_nodes)
    ]

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=320,
        alpha=0.95,
        ax=ax
    )

    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    sm = cm.ScalarMappable(cmap=cm.Blues, norm=mcolors.Normalize(vmin=0, vmax=max_val))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Explainer Node Importance', rotation=270, labelpad=15)

    #ax.set_title(
    #    f'{dataset_name} | {model_name} | {explainer_name}',
    #    fontsize=12,
    #    fontweight='bold'
    #)
    ax.axis('off')

    plt.tight_layout()

    filename = f'{dataset_name}_{model_name}_{explainer_name}_graph{graph_idx}.pdf'
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_graph_explanation(edge_index, gt_node_mask, explain_node_mask, 
                                 dataset_name, model_name, explainer_name, 
                                 graph_idx, target_viz, save_dir='figures_viz'):
    """Create visualization showing full graph, ground truth motif, and explanation."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create networkx graph
    G = nx.Graph()
    num_nodes = max(edge_index.max().item() + 1, len(gt_node_mask))
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.cpu().t().numpy()
    G.add_edges_from(edges)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    # Plot 1: Ground Truth Motif
    ax = axes[0]
    node_colors_gt = ['red' if gt_node_mask[i] == 1 else 'lightgray' for i in range(num_nodes)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_gt, node_size=300, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_title(f'Ground Truth Motif\n{dataset_name}', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Plot 2: Explainer Output
    ax = axes[1]
    # Normalize explain_node_mask for visualization
    explain_mask_viz = explain_node_mask.detach().cpu().numpy() if isinstance(explain_node_mask, torch.Tensor) else explain_node_mask
    max_val = explain_mask_viz.max() if explain_mask_viz.max() > 0 else 1
    node_colors_exp = [cm.Reds(explain_mask_viz[i] / max_val) if explain_mask_viz[i] > 0 
                       else (0.9, 0.9, 0.9, 1.0) for i in range(num_nodes)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_exp, node_size=300, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_title(f'Explanation: {explainer_name}\nModel: {model_name}', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Plot 3: Overlay comparison
    ax = axes[2]
    node_colors_overlap = []
    for i in range(num_nodes):
        if gt_node_mask[i] == 1 and explain_mask_viz[i] > 0.5:
            node_colors_overlap.append('green')  # Correct
        elif gt_node_mask[i] == 1:
            node_colors_overlap.append('red')  # Missed (False Negative)
        elif explain_mask_viz[i] > 0.5:
            node_colors_overlap.append('orange')  # False Positive
        else:
            node_colors_overlap.append('lightgray')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_overlap, node_size=300, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_title('Overlap Analysis\n(Green=Match, Red=Missed, Orange=FP)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    filename = f'{dataset_name}_{model_name}_{explainer_name}_graph{graph_idx}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def save_explanation_details(edge_index, gt_node_mask, explain_node_mask, 
                             dataset_name, model_name, explainer_name, 
                             graph_idx, target_viz, metrics_dict, 
                             save_dir='llm_input'):
    """Save detailed explanation information to a text file."""
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f'{dataset_name}_{model_name}_{explainer_name}_graph{graph_idx}.txt'
    filepath = os.path.join(save_dir, filename)
    
    # Compute graph statistics
    num_nodes = max(edge_index.max().item() + 1, len(gt_node_mask))
    triangles = count_triangles_per_node(edge_index, num_nodes)
    
    # Create networkx graph for additional metrics
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.cpu().t().numpy()
    G.add_edges_from(edges)
    
    degrees = dict(G.degree())
    clustering_coeffs = nx.clustering(G)
    closeness = nx.closeness_centrality(G)
    
    # Identify nodes
    gt_nodes = [i for i in range(num_nodes) if gt_node_mask[i] == 1]
    explain_mask_viz = explain_node_mask.detach().cpu().numpy() if isinstance(explain_node_mask, torch.Tensor) else explain_node_mask
    exp_nodes_top = np.argsort(explain_mask_viz)[::-1][:len(gt_nodes)].tolist()
    
    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"GRAPH EXPLANATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Explainer: {explainer_name}\n")
        f.write(f"Graph Index: {graph_idx}\n")
        #f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prediction (sigmoid): {target_viz.item():.4f}\n")
        f.write("\n" + "-" * 80 + "\n\n")
        
        # Graph Statistics
        f.write("GRAPH STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of nodes: {num_nodes}\n")
        f.write(f"Number of edges: {edge_index.detach().cpu().numpy()}\n")
        f.write(f"Number of edges: {edge_index.shape[1]}\n")
        f.write(f"Average degree: {sum(degrees.values()) / len(degrees):.2f}\n")
        f.write(f"Average clustering coefficient: {sum(clustering_coeffs.values()) / len(clustering_coeffs):.4f}\n")
        f.write(f"Average closeness centrality: {sum(closeness.values()) / len(closeness):.4f}\n")
        f.write(f"Total triangles in graph: {sum(triangles.values()) // 3}\n")
        f.write("\n" + "-" * 80 + "\n\n")
        
        # Per-Node Feature Vectors
        f.write("PER-NODE FEATURE VECTORS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Degree vector (all nodes): {[degrees[i] for i in range(num_nodes)]}\n")
        f.write(f"Triangle count vector (all nodes): {[triangles[i] for i in range(num_nodes)]}\n")
        f.write(f"Closeness centrality vector (all nodes): {[closeness[i] for i in range(num_nodes)]}\n")
        f.write("\n" + "-" * 80 + "\n\n")
        
        # Explanation Metrics
        f.write("EXPLANATION METRICS\n")
        f.write("-" * 40 + "\n")
        if metrics_dict:
            for metric_name, metric_value in metrics_dict.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
        f.write("\n" + "-" * 80 + "\n\n")
        
        # Ground Truth Motif Analysis
        f.write("GROUND TRUTH MOTIF\n")
        f.write("-" * 40 + "\n")
        f.write(f"Ground truth nodes: {gt_nodes}\n")
        f.write(f"Number of nodes in motif: {len(gt_nodes)}\n\n")
        
        f.write("Node Details:\n")
        f.write(f"{'Node':<8} {'Degree':<10} {'Triangles':<12} {'Clustering':<12} {'Closeness':<12}\n")
        f.write("-" * 56 + "\n")
        for node in gt_nodes:
            f.write(f"{node:<8} {degrees[node]:<10} {triangles[node]:<12} {clustering_coeffs[node]:<12.4f} {closeness[node]:<12.4f}\n")
        
        f.write("\n" + "-" * 80 + "\n\n")
        
        # Explanation Analysis
        f.write("EXPLAINER OUTPUT\n")
        f.write("-" * 40 + "\n")
        f.write(f"Top-{len(gt_nodes)} explanation nodes: {exp_nodes_top}\n")
        f.write(f"Number of predicted important nodes: {len(exp_nodes_top)}\n\n")
        
        f.write("Node Details (sorted by importance):\n")
        f.write(f"{'Node':<8} {'Score':<12} {'Degree':<10} {'Triangles':<12} {'Clustering':<12} {'Closeness':<12}\n")
        f.write("-" * 68 + "\n")
        for node in exp_nodes_top:
            score = explain_mask_viz[node]
            f.write(f"{node:<8} {score:<12.4f} {degrees[node]:<10} {triangles[node]:<12} {clustering_coeffs[node]:<12.4f} {closeness[node]:<12.4f}\n")
        
        f.write("\n" + "-" * 80 + "\n\n")
        
        # Overlap Analysis
        f.write("OVERLAP ANALYSIS\n")
        f.write("-" * 40 + "\n")
        correct_nodes = [n for n in exp_nodes_top if n in gt_nodes]
        missed_nodes = [n for n in gt_nodes if n not in exp_nodes_top]
        false_pos_nodes = [n for n in exp_nodes_top if n not in gt_nodes]
        
        f.write(f"Correctly identified nodes: {correct_nodes}\n")
        f.write(f"Missed ground truth nodes: {missed_nodes}\n")
        f.write(f"False positive nodes: {false_pos_nodes}\n\n")
        
        f.write(f"True Positives: {len(correct_nodes)}\n")
        f.write(f"False Negatives: {len(missed_nodes)}\n")
        f.write(f"False Positives: {len(false_pos_nodes)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    return filepath


def evaluate_gc_explainer_on_data(explainer, data_list, metric_names, use_prob=False, gt_metrics=None, threshold=0.5, 
                                  dataset_name=None, model_name=None, explainer_name=None, save_viz=False):
    eval_metrics = {metric_name: 0 for metric_name in metric_names}
    if gt_metrics is None:
        gt_metrics = ['accuracy', 'auroc']
    accs, precisions, recalls, ious, pos_fids, neg_fids, us, inf_times = [], [], [], [], [], [], [], []
    
    graph_idx = 0
    for data in data_list:
        start = time.time()
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=explainer.model(data.x, data.edge_index) if use_prob else data.y.unsqueeze(0)
        )
        end = time.time()
        inference_time = end - start
        inf_times.append(inference_time)
        data_nodes = data.edge_index[:, data.edge_mask.bool()].view(-1).unique()
        data_node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device, requires_grad=False)
        data_node_mask[data_nodes] = 1

        if 'node_mask' in explanation:
            node_mask = explanation.node_mask
        else:
            nodes = explanation.edge_index[:, explanation.edge_mask.bool()].view(-1).unique()
            node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device)
            node_mask[nodes] = 1

        explanation.edge_mask = explanation.edge_mask.float() if explanation.edge_mask.dtype == torch.bool else explanation.edge_mask
        gt_nodes = (torch.nonzero(data_node_mask.view(-1) == 1).squeeze()).detach().cpu()
        exp_nodes = (node_mask.view(-1).sort(descending=True)[1][:gt_nodes.size(0)]).detach().cpu()
        node_pred_mask = torch.zeros_like(node_mask.view(-1))
        node_pred_mask[exp_nodes] = 1

        #viz

        edge_index_viz = data.edge_index
        target_viz = torch.sigmoid(explainer.model(data.x, data.edge_index))
        gt_node_mask = data_node_mask.view(-1).cpu()
        gt_edge_mask = data.edge_index[:, data.edge_mask.bool()]
        explain_node_mask = node_mask.view(-1).cpu()
        filtered_explain_mask = torch.zeros_like(explain_node_mask)
        filtered_explain_mask[exp_nodes] = explain_node_mask[exp_nodes]
        explain_node_mask = filtered_explain_mask
        
        # Create visualizations and save detailed information if context provided
        if save_viz and dataset_name and model_name and explainer_name:
            # Compute current metrics for this graph
            current_metrics = {
                'iou': custom_iou(gt_nodes, exp_nodes).item() if hasattr(custom_iou(gt_nodes, exp_nodes), 'item') else custom_iou(gt_nodes, exp_nodes),
                'accuracy': groundtruth_metrics(node_pred_mask, data_node_mask.view(-1), metrics=['accuracy'], threshold=0.5),
            }
            
            # Generate visualization
            viz_path = visualize_graph_explanation_single(
                edge_index=edge_index_viz, gt_edge_mask=gt_edge_mask,gt_node_mask=gt_node_mask, explain_node_mask=explain_node_mask,
                dataset_name=dataset_name, model_name=model_name, explainer_name=explainer_name,
                graph_idx=graph_idx, save_dir='figures_viz'
            )
            
            # Save detailed text information
            text_path = save_explanation_details(
                edge_index_viz, gt_node_mask, explain_node_mask,
                dataset_name, model_name, explainer_name,
                graph_idx, target_viz, current_metrics
            )
            if graph_idx >= 20:
                save_viz = False  # Limit to first 20 graphs for visualization and detailed saving

        #################
        
        graph_idx += 1

        #node_pred_mask = torch.zeros_like(node_mask.view(-1))
        #node_pred_mask[exp_nodes] = 1
        iou = custom_iou(gt_nodes, exp_nodes)
        precision = custom_precision(gt_nodes, exp_nodes)
        recall = custom_recall(gt_nodes, exp_nodes)
        gt, pred = data_node_mask.view(-1), node_pred_mask.view(-1)
        acc = groundtruth_metrics(pred, gt, metrics=['accuracy'], threshold=0.5)
        # auroc = groundtruth_metrics(pred, gt, metrics=['auroc'])
        # iou = binary_jaccard_index(pred, gt, threshold=threshold).cpu()
        pos_fidelity, neg_fidelity = custom_fidelity(explainer, explanation, node_mask, max_nodes=gt_nodes.size(0), full_graph=True)
        u = unfaithfulness(explainer, explanation)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        ious.append(iou)
        pos_fids.append(pos_fidelity)
        neg_fids.append(neg_fidelity)
        us.append(u)

    eval_metrics['accuracy'] = np.mean(accs)
    eval_metrics['precision'] = np.mean(precisions)
    eval_metrics['recall'] = np.mean(recalls)
    eval_metrics['iou'] = np.mean(ious)
    eval_metrics['fid+'] = np.mean(pos_fids)
    eval_metrics['fid-'] = np.mean(neg_fids)
    eval_metrics['unfaithfulness'] = np.mean(us)
    eval_metrics['inference_time'] = np.mean(inf_times)
    eval_metrics['characterization_score'] = characterization_score(eval_metrics['fid+'], eval_metrics['fid-'])
    return eval_metrics


def evaluate_gc_explainer(model_path, explainer_name, explainer_config, gc_datasets, metric_names, std=None,save_viz=True):
    start_time = time.time()

    print(f'{"-" * 2} Evaluating {explainer_name} explainer on graph classification datasets...')
    exp_eval_metrics = {}
    model_config = ModelConfig(mode='binary_classification', task_level='graph', return_type='raw')
    for dataset_name, test_data_list, val_data_list in gc_datasets:
        gc_dataset_start_time = time.time()
        print(f'{"-" * 3} Evaluating {explainer_name} explainer on {dataset_name} dataset...')

        gc_models = get_gnn(model_path, 'gc', 'all', dataset_name, std='none' if std is None else std)
        gc_models = setup_models(gc_models, test_data_list[0].x.device)

        for model_name, model in gc_models:
            gc_model_start_time = time.time()
            model_name = f'{model_name}-{std}'
            print(f'{"-" * 5} Evaluating {explainer_name} explainer on {model_name} model...')
            if (explainer_name, model_name) not in exp_eval_metrics:
                exp_eval_metrics[(explainer_name, model_name)] = {}

            use_prob = explainer_name == 'ciexplainer'

            cat_feat_indices = None
            if dataset_name == 'mutag':
                cat_feat_indices = [0]
            explainer = get_explainer(explainer_name, explainer_config, model, model_config, dataset=test_data_list,
                                      cat_feat_indices=cat_feat_indices, dataset_name=dataset_name,val_dataset=val_data_list)
            threshold = 0.5
            if explainer_name == 'ciexplainer':
                threshold = 0.0
            res = evaluate_gc_explainer_on_data(explainer, test_data_list, metric_names, use_prob, gt_metrics=None,
                                                threshold=threshold, dataset_name=dataset_name, 
                                                model_name=model_name, explainer_name=explainer_name, 
                                                save_viz=save_viz)
            for metric_name, metric_value in res.items():
                exp_eval_metrics[(explainer_name, model_name)][(dataset_name, metric_name)] = metric_value

            gc_model_end_time = time.time()
            gc_mode_elapsed_time = (gc_model_end_time - gc_model_start_time) / 60
            print(f'{"-" * 7} Evaluation on {model_name} model took {gc_mode_elapsed_time:.2f} minutes.')

        gc_dataset_end_time = time.time()
        gc_dataset_elapsed_time = (gc_dataset_end_time - gc_dataset_start_time) / 60
        print(f'{"-" * 6} Evaluation on {dataset_name} dataset took {gc_dataset_elapsed_time:.2f} minutes.')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f'{"-" * 3} Evaluation on graph classification took {elapsed_time:.2f} minutes.')
    return exp_eval_metrics

# def evaluate_gc_explainer_on_data(explainer, data_list, metric_names, use_prob=False, gt_metrics=None, threshold=0.5):
#     eval_metrics = {metric_name: 0 for metric_name in metric_names}
#     if gt_metrics is None:
#         gt_metrics = ['accuracy', 'auroc']
#     accs, aurocs, ious, pos_fids, neg_fids, us, inf_times = [], [], [], [], [], [], []
#     for data in data_list:
#         start = time.time()
#         explanation = explainer(
#             x=data.x,
#             edge_index=data.edge_index,
#             target=explainer.model(data.x, data.edge_index).sigmoid() if use_prob else data.y.unsqueeze(0)
#         )
#         end = time.time()
#         inference_time = end - start
#         inf_times.append(inference_time)
#         data_nodes = data.edge_index[:, data.edge_mask.bool()].view(-1).unique()
#         data_node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device, requires_grad=False)
#         data_node_mask[data_nodes] = 1
#
#         if 'node_mask' in explanation:
#             node_mask = explanation.node_mask
#         else:
#             nodes = explanation.edge_index[:, explanation.edge_mask.bool()].view(-1).unique()
#             node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device)
#             node_mask[nodes] = 1
#         explanation.edge_mask = explanation.edge_mask.float() if explanation.edge_mask.dtype == torch.bool else explanation.edge_mask
#         gt, pred = data_node_mask.view(-1), node_mask.view(-1)
#         acc = groundtruth_metrics(pred, gt, metrics=['accuracy'], threshold=threshold)
#         auroc = groundtruth_metrics(pred, gt, metrics=['auroc'])
#         iou = binary_jaccard_index(pred, gt, threshold=threshold).cpu()
#         pos_fidelity, neg_fidelity = fidelity(explainer, explanation)
#         u = unfaithfulness(explainer, explanation)
#         accs.append(acc)
#         aurocs.append(auroc)
#         ious.append(iou)
#         pos_fids.append(pos_fidelity)
#         neg_fids.append(neg_fidelity)
#         us.append(u)
#
#     eval_metrics['accuracy'] = np.mean(accs)
#     eval_metrics['auroc'] = np.mean(aurocs)
#     eval_metrics['iou'] = np.mean(ious)
#     eval_metrics['fid+'] = np.mean(pos_fids)
#     eval_metrics['fid-'] = np.mean(neg_fids)
#     eval_metrics['unfaithfulness'] = np.mean(us)
#     eval_metrics['inference_time'] = np.mean(inf_times)
#     eval_metrics['characterization_score'] = characterization_score(eval_metrics['fid+'], eval_metrics['fid-'])
#     return eval_metrics
#
#
# def evaluate_gc_explainer(model_path, explainer_name, explainer_config, gc_datasets, metric_names):
#     start_time = time.time()
#
#     print(f'{"-" * 2} Evaluating {explainer_name} explainer on graph classification datasets...')
#     exp_eval_metrics = {}
#     model_config = ModelConfig(mode='binary_classification', task_level='graph', return_type='raw')
#     for dataset_name, test_data_list, num_classes in gc_datasets:
#         gc_dataset_start_time = time.time()
#
#         print(f'{"-" * 3} Evaluating {explainer_name} explainer on {dataset_name} dataset...')
#         gc_models = get_gnn(model_path, 'gc', 'all', dataset_name)
#         gc_models = setup_models(gc_models, test_data_list[0].x.device)
#
#         for model_name, model in gc_models:
#             gc_model_start_time = time.time()
#
#             print(f'{"-" * 5} Evaluating {explainer_name} explainer on {model_name} model...')
#             if (explainer_name, model_name) not in exp_eval_metrics:
#                 exp_eval_metrics[(explainer_name, model_name)] = {}
#
#             use_prob = explainer_name == 'ciexplainer'
#
#             cat_feat_indices = None
#             if dataset_name == 'mutag':
#                 cat_feat_indices = [0]
#             explainer = get_explainer(explainer_name, explainer_config, model, model_config, dataset=test_data_list,
#                                       dataset_name=dataset_name, cat_feat_indices=cat_feat_indices)
#             threshold = 0.5
#             if explainer_name == 'ciexplainer':
#                 threshold = 0.0
#             res = evaluate_gc_explainer_on_data(explainer, test_data_list, metric_names, use_prob, gt_metrics=None,
#                                                 threshold=threshold)
#             for metric_name, metric_value in res.items():
#                 exp_eval_metrics[(explainer_name, model_name)][(dataset_name, metric_name)] = metric_value
#
#             gc_model_end_time = time.time()
#             gc_mode_elapsed_time = (gc_model_end_time - gc_model_start_time) / 60
#             print(f'{"-" * 7} Evaluation on {model_name} model took {gc_mode_elapsed_time:.2f} minutes.')
#
#         gc_dataset_end_time = time.time()
#         gc_dataset_elapsed_time = (gc_dataset_end_time - gc_dataset_start_time) / 60
#         print(f'{"-" * 6} Evaluation on {dataset_name} dataset took {gc_dataset_elapsed_time:.2f} minutes.')
#
#     end_time = time.time()
#     elapsed_time = (end_time - start_time) / 60
#     print(f'{"-" * 3} Evaluation on graph classification took {elapsed_time:.2f} minutes.')
#     return exp_eval_metrics
