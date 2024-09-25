from pynvml import *
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import pandas as pd
import os 

import dgl
import torch
import torch.nn.functional as F
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    negative_sampler,
    NeighborSampler,
)

from sklearn.metrics import confusion_matrix
import itertools

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

from model import SAGE

#========== GPU memory checker  

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used // 1024**2

#========= Plot Functions

def plot_metrics(train_metric_values, test_metric_values, metric_name, args):
    epochs = range(1, args.epochs + 1)
    plt.figure()
    plt.plot(epochs, train_metric_values, label=f'Train', color='darkblue', marker='o')
    plt.plot(epochs, test_metric_values, label=f'Test', color='red', marker='x')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    #plt.title(f'{metric_name} over Epochs')
    plt.xticks(epochs)  # Set x-axis ticks to integer values of epochs
    plt.legend()
    plt.savefig(f'{args.output_folder}/{metric_name.lower()}.png')

def plot_roc_curve(train_labels, train_scores, test_labels, test_scores, args):
    
    train_fpr, train_tpr, _ = roc_curve(train_labels, train_scores)
    test_fpr, test_tpr, _ = roc_curve(test_labels, test_scores)

    plt.figure()
    plt.plot(train_fpr, train_tpr, color='blue', lw=2, label='Train ROC curve')
    plt.plot(test_fpr, test_tpr, color='darkorange', lw=2, label='Test ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.grid(False)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f'{args.output_folder}/roc_curve.png')

#=============  DGL graph handling

def to_bidirected_with_reverse_mapping(g):
    
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts="count", writeback_mapping=True
    )

    c = g_simple.edata["count"]
    num_edges = g.num_edges()

    mapping_offset = torch.zeros(
        g_simple.num_edges() + 1, dtype=g_simple.idtype
    )

    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]

    reverse_idx = torch.where(
        idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges
    )
    reverse_mapping = mapping[reverse_idx]

    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)

    return g_simple, reverse_mapping

#======= Train Test Split

def create_train_test_graphs(original_graph, bridge_edges_arg, test_ratio=0.3):
    feat = original_graph.ndata['feat']
    num_nodes = original_graph.number_of_nodes()
    all_edges = set(zip(original_graph.edges()[0].numpy(), original_graph.edges()[1].numpy()))
    print(f"Total number of edges in original graph: {len(all_edges)}")

    if isinstance(bridge_edges_arg, str):
        try:
            with open(bridge_edges_arg, 'rb') as file:
                bridge_edges = pickle.load(file)
            
            print(len(all_edges.intersection(bridge_edges)))
            non_bridge_edges = all_edges - set(bridge_edges)
            print(f"Number of bridge edges loaded: {len(bridge_edges)}")
            print(f"Number of bridge edges in original graph: {len(all_edges.intersection(bridge_edges))}")
            print(f"Number of non-bridge edges: {len(non_bridge_edges)}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Bridge edges file {bridge_edges_arg} not found.")
    elif isinstance(bridge_edges_arg, int):
        non_bridge_edges = all_edges
        bridge_edges = []
        print("No bridge edges provided.")
    else:
        raise ValueError("Invalid value for bridge_edges argument.")

    
    non_bridge_edges = list(non_bridge_edges)
    np.random.shuffle(non_bridge_edges)
    split_point = int(len(non_bridge_edges) * (1 - test_ratio))

    train_edges = non_bridge_edges[:split_point]
    print(f"Number of edges allocated to train set before adding bridge edges: {len(train_edges)}")

    if bridge_edges:
        train_edges += bridge_edges
        print(f"Adding {len(bridge_edges)} bridge edges to train set.")

    test_edges = non_bridge_edges[split_point:]
    
    train_edges = tuple(zip(*train_edges))
    test_edges = tuple(zip(*test_edges))

    train_graph = dgl.graph(train_edges, num_nodes=num_nodes)
    train_graph.ndata['feat'] = feat

    test_graph = dgl.graph(test_edges, num_nodes=num_nodes)
    test_graph.ndata['feat'] = feat

    print(f"Final number of edges in train graph after adding bridge edges: {train_graph.number_of_edges()}")

    total_edges_after_split = train_graph.number_of_edges() + test_graph.number_of_edges()
    print(f"Total edges after split: {total_edges_after_split}")
    if total_edges_after_split == len(all_edges):
        print("Edge counts match.")
    else:
        print(f"Discrepancy in edge counts: Expected {len(all_edges)}, got {total_edges_after_split}")


    return train_graph, test_graph

#======= Training loop

def train(args, device, train_set, test_set, model, use_uva, fused_sampling, validation = None):

    train_g, train_reverse_eids, train_seed_edges = train_set
    
    train_edges = set(zip(train_g.edges()[0].numpy(), train_g.edges()[1].numpy()))
    
    train_sampler = NeighborSampler(
        [15, 10, 5],
        prefetch_node_feats = ["feat"],
        fused = fused_sampling,
    )
    train_sampler = as_edge_prediction_sampler(
        train_sampler,
        exclude = "reverse_id" if args.exclude_edges else None,
        reverse_eids = train_reverse_eids if args.exclude_edges else None,
        negative_sampler=negative_sampler.Uniform(1),
    )

    train_dataloader = DataLoader(
        train_g,
        train_seed_edges,
        train_sampler,
        device=device,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    print('Training dataloader ready!')

    test_g, test_reverse_eids, test_seed_edges = test_set 
    
    test_edges = set(zip(test_g.edges()[0].numpy(), test_g.edges()[1].numpy()))

    test_sampler = NeighborSampler(
        [15, 10, 5],
        prefetch_node_feats = ["feat"],
        fused = fused_sampling,
    )
    test_sampler = as_edge_prediction_sampler(
        test_sampler,
        exclude = "reverse_id" if args.exclude_edges else None,
        reverse_eids = test_reverse_eids if args.exclude_edges else None,
        negative_sampler=negative_sampler.Uniform(1),
    )

    test_dataloader = DataLoader(
        test_g,
        test_seed_edges,
        test_sampler,
        device=device,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    print('Test dataloader ready!')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train_epoch_accuracies = []
    train_epoch_sensitivities = []
    train_epoch_specificities = []
    train_epoch_precisions = []
    train_epoch_f1s = []
    train_epoch_recalls = []
    train_losses = []
    train_labels = []
    train_scores = []
    train_roc_aucs = []


    test_epoch_accuracies = []
    test_epoch_sensitivities = []
    test_epoch_specificities = []
    test_epoch_precisions = []
    test_epoch_f1s = []
    test_epoch_recalls = []
    test_losses = []
    test_labels = []
    test_scores = []
    test_roc_aucs = [] 

    print('Start training!')
    print(model)
    for epoch in range(args.epochs):

        model.train()
        total_loss = 0
        start_epoch_time = time.time()

        train_accuracy = 0
        train_sensitivity = 0
        train_specificity = 0
        train_precision = 0
        train_f1 = 0
        train_recall = 0
        train_loss = 0

        train_num_batches = 0

        test_accuracy = 0
        test_sensitivity = 0
        test_specificity = 0
        test_precision = 0
        test_f1 = 0
        test_recall = 0
        test_loss = 0
        test_num_batches = 0

        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]

            neg_examples = set(zip(neg_pair_graph.edges()[0].cpu().numpy(), neg_pair_graph.edges()[1].cpu().numpy()))
            overlap = neg_examples.intersection(test_edges)
            
            non_overlapping_indices = [i for i, e in enumerate(neg_examples) if e not in overlap]
            neg_pair_graph = neg_pair_graph.edge_subgraph(non_overlapping_indices)

            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
            score = torch.cat([pos_score, neg_score])
            
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])

            loss = F.binary_cross_entropy_with_logits(score, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

            predictions = torch.round(torch.sigmoid(score)).cpu().detach().numpy()
            labels = labels.cpu().numpy()

            train_accuracy += accuracy_score(labels, predictions)
            train_sensitivity += recall_score(labels, predictions)
            train_specificity += recall_score(labels, predictions, pos_label=0)
            train_precision += precision_score(labels, predictions)
            train_f1 += f1_score(labels, predictions)
            train_recall += recall_score(labels, predictions)

            train_scores.extend(score.detach().cpu().numpy())
            train_labels.extend(labels)
            
            train_num_batches += 1
        
        with torch.no_grad():
           model.eval()

           for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
            test_dataloader
            ):

            neg_examples = set(zip(neg_pair_graph.edges()[0].cpu().numpy(), neg_pair_graph.edges()[1].cpu().numpy()))

            overlap = neg_examples.intersection(train_edges)
            
            non_overlapping_indices = [i for i, e in enumerate(neg_examples) if e not in overlap]
            neg_pair_graph = neg_pair_graph.edge_subgraph(non_overlapping_indices)

            x = blocks[0].srcdata["feat"]
            
            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
            score = torch.cat([pos_score, neg_score])

            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])

            loss = F.binary_cross_entropy_with_logits(score, labels)
            test_loss += loss.item()

            predictions = torch.round(torch.sigmoid(score)).cpu().numpy()
            labels = labels.cpu().numpy()

            test_accuracy += accuracy_score(labels, predictions)
            test_sensitivity += recall_score(labels, predictions)
            test_specificity += recall_score(labels, predictions, pos_label=0)
            test_precision += precision_score(labels, predictions)
            test_f1 += f1_score(labels, predictions)
            test_recall += recall_score(labels, predictions)

            test_scores.extend(score.detach().cpu().numpy())
            test_labels.extend(labels)
            
            test_num_batches += 1

        train_roc_auc = roc_auc_score(train_labels, train_scores)
        test_roc_auc = roc_auc_score(test_labels, test_scores)
        
        train_roc_aucs.append(train_roc_auc)
        test_roc_aucs.append(test_roc_auc)

        train_epoch_accuracies.append(train_accuracy / train_num_batches)
        train_epoch_sensitivities.append(train_sensitivity / train_num_batches)
        train_epoch_specificities.append(train_specificity / train_num_batches)
        train_epoch_precisions.append(train_precision / train_num_batches)
        train_epoch_f1s.append(train_f1 / train_num_batches)
        train_epoch_recalls.append(train_recall / train_num_batches)
        train_losses.append(train_loss / train_num_batches)

        test_epoch_accuracies.append(test_accuracy / test_num_batches)
        test_epoch_sensitivities.append(test_sensitivity / test_num_batches)
        test_epoch_specificities.append(test_specificity / test_num_batches)
        test_epoch_precisions.append(test_precision / test_num_batches)
        test_epoch_f1s.append(test_f1 / test_num_batches) 
        test_epoch_recalls.append(test_recall / test_num_batches)
        test_losses.append(test_loss / test_num_batches)

        end_epoch_time = time.time()

        print(f"\nEpoch {epoch:05d} Time:{(end_epoch_time - start_epoch_time):.4f}s \n"
              f"\nTraining Data: \nLoss: {train_loss / train_num_batches:.4f}"
              f"\nAccuracy: {train_accuracy / train_num_batches:.4f} \nSensitivity: {train_sensitivity / train_num_batches:.4f}"
              f"\nSpecificity: {train_specificity / train_num_batches:.4f} \nPrecision: {train_precision / train_num_batches:.4f}"
              f"\nF1 Score: {train_f1 / train_num_batches:.4f} \nRecall: {train_recall / train_num_batches:.4f}"
              f"\nROC AUC: {train_roc_auc:.4f}"  # Print training ROC AUC
              f"\nTest Data: \nLoss: {test_loss / test_num_batches:.4f}"
              f"\nAccuracy: {test_accuracy / test_num_batches:.4f} \nSensitivity: {test_sensitivity / test_num_batches:.4f}"
              f"\nSpecificity: {test_specificity / test_num_batches:.4f} \nPrecision: {test_precision / test_num_batches:.4f}"
              f"\nF1 Score: {test_f1 / test_num_batches:.4f} \nRecall: {test_recall / test_num_batches:.4f}"
              f"\nROC AUC: {test_roc_auc:.4f}"  # Print test ROC AUC
              )


    

    test_predictions = torch.round(torch.sigmoid(torch.tensor(test_scores))).cpu().detach().numpy()
    conf_matrix = confusion_matrix(test_labels, test_predictions)

    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])

    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{args.output_folder}/confusion_matrix.png')
    plt.close()

    plot_metrics(train_losses, test_losses, 'Loss (BCE with logits)', args)
    plot_metrics(train_epoch_accuracies, test_epoch_accuracies, 'Accuracy', args)
    plot_metrics(train_epoch_sensitivities, test_epoch_sensitivities, 'Sensitivity', args)
    plot_metrics(train_epoch_specificities, test_epoch_specificities, 'Specificity', args)
    plot_metrics(train_epoch_precisions, test_epoch_precisions, 'Precision', args)
    plot_metrics(train_epoch_f1s, test_epoch_f1s, 'F1 Score', args)
    plot_metrics(train_epoch_recalls, test_epoch_recalls, 'Recall', args)
    plot_roc_curve(train_labels, train_scores, test_labels, test_scores, args)
    
    train_metrics = {
        "Accuracy": train_epoch_accuracies,
        "Sensitivity": train_epoch_sensitivities,
        "Specificity": train_epoch_specificities,
        "Precision": train_epoch_precisions,
        "F1": train_epoch_f1s,
        "Recall": train_epoch_recalls,
        "Loss": train_losses,
        "ROC AUC": train_roc_aucs,  # Save training ROC AUC values
    }

    test_metrics = {
        "Accuracy": test_epoch_accuracies,
        "Sensitivity": test_epoch_sensitivities,
        "Specificity": test_epoch_specificities,
        "Precision": test_epoch_precisions,
        "F1": test_epoch_f1s,
        "Recall": test_epoch_recalls,
        "Loss": test_losses,
        "ROC AUC": test_roc_aucs,  # Save test ROC AUC values
    }
    
    train_metrics_df = pd.DataFrame(train_metrics)
    test_metrics_df = pd.DataFrame(test_metrics)

    metrics_dir = os.path.join(args.output_folder, 'metrics')
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    train_metrics_df.to_csv(os.path.join(metrics_dir, 'train_metrics.csv'), index=False)
    test_metrics_df.to_csv(os.path.join(metrics_dir, 'test_metrics.csv'), index=False)

    print("Metrics saved!")


def str_or_int(value):
    try:
        return int(value)
    except ValueError:
        return value

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-graph", type=str)
    parser.add_argument("--bridge-edges", type=str_or_int, default=0,
                        help="Path to the file with bridge edges or a number to exclude bridge edges. Default: 0 (exclude)")

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test data size. Default: 0.2",
    )

    parser.add_argument("--epochs", type=int, default=10) 
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate. Default: 0.0005",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Batch size for training. Default: 256",
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=512,
        help="Batch size for training. Default: 512",
    )

    parser.add_argument(
        "--exclude-edges",
        type=int,
        default=1,
        help="Whether to exclude reverse edges during sampling. Default: 1",
    ) 

    parser.add_argument("--output-folder", type=str)

    return parser.parse_args()

def main(args):

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    else:
        print(f"Directory {args.output_folder} already exists. Using the existing directory.")

    print("Loading data...")
    g, _ = dgl.load_graphs(args.training_graph)
    g = g[0]    

    print(g.number_of_edges())

    print('Done')

    train_g, test_g = create_train_test_graphs(g, args.bridge_edges, test_ratio=args.test_size)
    
    print(f'Train set number of edges {train_g.number_of_edges()}')
    print(f'Test set number of edges {test_g.number_of_edges()}')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Device for training: {device} --> {torch.cuda.get_device_name(0)}')
        print(f'GPU memory occupied before initialization: {print_gpu_utilization()} MB')
    else:
        device = torch.device("cpu")

    train_g, train_reverse_eids = to_bidirected_with_reverse_mapping(train_g)
    train_reverse_eids = train_reverse_eids.to(g.device)
    train_seed_edges = torch.arange(train_g.num_edges()).to(g.device)

    train_set = [train_g, train_reverse_eids, train_seed_edges]

    test_g, test_reverse_eids = to_bidirected_with_reverse_mapping(test_g)
    test_reverse_eids = test_reverse_eids.to(g.device)
    test_seed_edges = torch.arange(test_g.num_edges()).to(g.device)

    test_set = [test_g, test_reverse_eids, test_seed_edges]

    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, args.hidden_size).to(device)
    print(f'Model has {sum(p.numel() for p in model.parameters())} parameters')

    if torch.cuda.is_available():
        print(f'GPU memory occupied after initialization: {print_gpu_utilization()} MB')

    train(
        args,
        device,
        train_set,
        test_set,
        model,
        use_uva = False,
        fused_sampling = False,
        validation = None
    )

    g, _ = to_bidirected_with_reverse_mapping(g)

    node_emb = model.inference(g, device, args.train_batch_size)
    node_emb = node_emb.detach()
    node_emb = node_emb.cpu().numpy()
    np.save(f'{args.output_folder}/embeddings.npy', node_emb)

    torch.save(model.state_dict(), f'{args.output_folder}/model.pth')

    print('Done')

if __name__ == "__main__":
    args = parse_args()
    main(args)
