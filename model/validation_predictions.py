import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dgl
from model import SAGE  

def load_pairs_from_csv(csv_path):
    """
    Load pairs and their corresponding confidence scores from a CSV file.
    """
    df = pd.read_csv(csv_path)
    pairs = set(zip(df['gene1'], df['gene2']))
    pairs |= set((b, a) for a, b in pairs)
    confidence_scores = df['combined_score'].values
    return pairs, confidence_scores

def load_model(model_path, in_size, hidden_size):
    """
    Load the trained GraphSAGE model from a file.
    """
    model = SAGE(in_size, hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode idiot!
    return model

def load_ucsc_pairs(ucsc_path):
    """
    Load UCSC dataset and return a set of pairs.
    """
    ucsc_df = pd.read_csv(ucsc_path)
    ucsc_pairs = set(zip(ucsc_df['gene1'], ucsc_df['gene2']))
    ucsc_pairs |= set((b, a) for a, b in ucsc_pairs)
    return ucsc_pairs

def filter_pairs_with_scores(pairs, confidence_scores, ucsc_pairs):
    """
    Filter out pairs that exist in the UCSC dataset and return corresponding confidence scores.
    """
    num_edges_before = len(pairs)
    common_pairs = pairs.intersection(ucsc_pairs)
    num_common_pairs = len(common_pairs)

    # Filter pairs that are not in UCSC dataset
    filtered_pairs = pairs - ucsc_pairs

    # Filter confidence scores to match filtered pairs
    filtered_scores = [score for pair, score in zip(pairs, confidence_scores) if pair in filtered_pairs]
    num_edges_after = len(filtered_pairs)

    print(f"Number of edges in STRING before filtering: {num_edges_before}")
    print(f"Number of edges common in UCSC and STRING: {num_common_pairs}")
    print(f"Number of edges in STRING after filtering: {num_edges_after}")

    return filtered_pairs, np.array(filtered_scores)

def create_dgl_graph_from_pairs(pairs, node_features):
    """
    Create a DGL graph from filtered pairs.
    """
    src, dst = zip(*pairs)
    graph = dgl.graph((src, dst))
    graph.ndata['feat'] = node_features
    return graph

def predict(graph, model):
    """
    Perform predictions on a given graph using the trained model.
    """
    with torch.no_grad():
        node_features = graph.ndata['feat']
        pair_graph = graph
        neg_pair_graph = dgl.graph(([], []), num_nodes=graph.number_of_nodes())
        blocks = [graph]  
        pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, node_features)
        return pos_score

def plot_violin_box_plots(predictions, label, title, output_path, file_name_prefix, y_axis_label, show_0_5_line=True):
    """
    Plot and save both a violin plot and a box plot for predictions.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    predictions = predictions.cpu().numpy()
    mean_score, max_score, min_score = np.mean(predictions), np.max(predictions), np.min(predictions)
    print(f"{label} - Mean: {mean_score}, Max: {max_score}, Min: {min_score}")

    plt.figure(figsize=(8, 6))
    sns.violinplot(y=predictions)
    if show_0_5_line:
        plt.axhline(0.5, linestyle='--', color='red')
        plt.legend()
    plt.ylabel(y_axis_label)
    plt.xlabel(label)
    plt.title(f'Violin Plot of {title}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{file_name_prefix}_violin_plot.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(y=predictions)
    if show_0_5_line:
        plt.axhline(0.5, linestyle='--', color='red')
    plt.ylabel(y_axis_label)
    plt.xlabel(label)
    plt.title(f'Box Plot of {title}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{file_name_prefix}_box_plot.png"))
    plt.close()

def categorize_and_summarize(predictions, confidence_scores):
    """
    Categorize predictions by confidence score ranges and summarize them.
    """
    print("Type of confidence_scores:", type(confidence_scores))
    print("Shape of confidence_scores:", confidence_scores.shape)
    print("First few confidence_scores:", confidence_scores[:10])

    categories = {
        "Very high (900-1000)": (900, 1000),
        "High (700-900)": (700, 900),
        "Medium (400-700)": (400, 700),
        "Low (200-400)": (200, 400),
        "Very low (0-200)": (0, 200)
    }

    summary = []

    for category, (low, high) in categories.items():
        print(f"Processing category: {category} with range {low}-{high}")
        indices = (confidence_scores >= low) & (confidence_scores < high)
        print("Type of indices:", type(indices))
        print("Indices shape:", indices.shape)
        category_probs = predictions[indices]
        mean_prob = category_probs.mean().item()
        above_0_5 = (category_probs > 0.5).sum().item()
        total = indices.sum().item()
        summary.append({
            "Confidence Level": category,
            "Number of Samples": total,
            "Mean Probability": mean_prob,
            "Probability > 0.5": f"{above_0_5} ({(above_0_5 / total) * 100:.2f}%)"
        })

    return pd.DataFrame(summary)

def main(csv_path, model_path, ucsc_path, in_size, hidden_size, plot_path, output_csv_path):
    string_pairs, confidence_scores = load_pairs_from_csv(csv_path)
    model = load_model(model_path, in_size, hidden_size)
    ucsc_pairs = load_ucsc_pairs(ucsc_path)
    filtered_pairs, filtered_confidence_scores = filter_pairs_with_scores(string_pairs, confidence_scores, ucsc_pairs)
    
    unique_nodes = set([gene for pair in filtered_pairs for gene in pair])
    node_to_id = {gene: i for i, gene in enumerate(unique_nodes)}
    node_features = torch.randn(len(unique_nodes), in_size)
    
    filtered_pairs_id = [(node_to_id[pair[0]], node_to_id[pair[1]]) for pair in filtered_pairs]

    graph = create_dgl_graph_from_pairs(filtered_pairs_id, node_features)

    pos_score = predict(graph, model).flatten()
    pos_score_sigmoid = torch.sigmoid(pos_score)
    
    print("Length of filtered_pairs_id:", len(filtered_pairs_id))
    print("Length of filtered_confidence_scores:", len(filtered_confidence_scores))
    
    summary_df = categorize_and_summarize(pos_score_sigmoid.cpu().numpy(), filtered_confidence_scores)
    
    summary_df.to_csv(output_csv_path, index=False)
    print(f"Summary saved to {output_csv_path}")

    plot_violin_box_plots(pos_score, "STRING scores", "String Scores (Raw)", plot_path, "positive_scores", "Scores", show_0_5_line=False)
    plot_violin_box_plots(pos_score_sigmoid, "physical STRING samples", "String Probabilities", plot_path, "positive_probabilities", "Probability")

if __name__ == "__main__":
    csv_path = '/home/angelosmath/MSc/thesis_final_last/mapping/STRING_gene.csv'
    model_path = '/home/angelosmath/MSc/thesis_final_last/model/run_kantale_last/model.pth'
    ucsc_path = '/home/angelosmath/MSc/thesis_final_last/data/ucsc/gg_ppi.csv'
    plot_path = '/home/angelosmath/MSc/thesis_final_last/model/validation'
    output_csv_path = '/home/angelosmath/MSc/thesis_final_last/model/validation/summary_table.csv'
    in_size = 200 
    hidden_size = 200
    main(csv_path, model_path, ucsc_path, in_size, hidden_size, plot_path, output_csv_path)
