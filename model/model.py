import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Three-layer GraphSAGE-mean.
        self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.hidden_size = hidden_size
        self.predictor = nn.Sequential(
        
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        hidden_pos = self.predictor(hidden_x[pos_src] * hidden_x[pos_dst])
        hidden_neg = self.predictor(hidden_x[neg_src] * hidden_x[neg_dst])

        # Compute dot product to get a single value for each edge
        #hidden_pos = (hidden_x[pos_src] * hidden_x[pos_dst]).sum(dim=1)
        #hidden_neg = (hidden_x[neg_src] * hidden_x[neg_dst]).sum(dim=1)

        
        return hidden_pos, hidden_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata["feat"]
        #####################################################################
        # (HIGHLIGHT) Creating a MultiLayerFullNeighborSampler instance.
        # This sampler is used in the Graph Neural Networks (GNN) training
        # process to provide neighbor sampling, which is crucial for
        # efficient training of GNN on large graphs.
        #
        # The first argument '1' indicates the number of layers for
        # the neighbor sampling. In this case, it's set to 1, meaning
        # only the direct neighbors of each node will be included in the
        # sampling.
        #
        # The 'prefetch_node_feats' parameter specifies the node features
        # that need to be pre-fetched during sampling. In this case, the
        # feature named 'feat' will be pre-fetched.
        #
        # `prefetch` in DGL initiates data fetching operations in parallel
        # with model computations. This ensures data is ready when the
        # computation needs it, thereby eliminating waiting times between
        # fetching and computing steps and reducing the I/O overhead during
        # the training process.
        #
        # The difference between whether to use prefetch or not is shown:
        #
        # Without Prefetch:
        # Fetch1 ──> Compute1 ──> Fetch2 ──> Compute2 ──> Fetch3 ──> Compute3
        #
        # With Prefetch:
        # Fetch1 ──> Fetch2 ──> Fetch3
        #    │          │          │
        #    └─Compute1 └─Compute2 └─Compute3
        #####################################################################
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        # Enable pin_memory for faster CPU to GPU data transfer if the model is
        # running on a GPU.
        pin_memory = buffer_device != device
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == len(self.layers) - 1
            y = torch.empty(
                g.num_nodes(),
                self.hidden_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc="Inference"
            ):
                x = feat[input_nodes]
                hidden_x = layer(blocks[0], x)
                if not is_last_layer:
                    hidden_x = F.relu(hidden_x)
                y[output_nodes] = hidden_x.to(buffer_device)
            feat = y
        return y
