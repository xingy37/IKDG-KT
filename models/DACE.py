import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class DACE(nn.Module):
    """
    DACE-style temporal encoder adapted to this project's dynamic link classification API.
    """

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        time_dim: int = 16,
        num_neighbors: int = 50,
        dropout: float = 0.1,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.device = device

        node_features = torch.from_numpy(node_raw_features.astype(np.float32))
        edge_features = torch.from_numpy(edge_raw_features.astype(np.float32))
        self.register_buffer("node_raw_features", node_features)
        self.register_buffer("edge_raw_features", edge_features)

        self.node_input_dim = node_features.shape[1]
        self.edge_input_dim = edge_features.shape[1]
        self.node_dim = 64
        self.time_dim = time_dim

        self.node_projection = nn.Linear(self.node_input_dim, self.node_dim)
        self.edge_projection = nn.Linear(self.edge_input_dim, self.node_dim)
        self.time_projection = nn.Linear(self.time_dim, self.node_dim)
        self.position_embedding = nn.Embedding(self.num_neighbors + 1, self.node_dim)

        self.time_encoder = TimeEncoder(time_dim=self.time_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.node_dim,
            nhead=4,
            dim_feedforward=2 * self.node_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(self.node_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(self.node_dim, 2 * self.node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.node_dim, self.node_dim),
        )
        self.output_norm = nn.LayerNorm(self.node_dim)
        self.readout = nn.Linear(2 * self.node_dim, self.node_dim)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    def compute_src_dst_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        edge_ids: np.ndarray,
        node_interact_times: np.ndarray,
        dst_node_ids: np.ndarray,
    ):
        _ = edge_ids
        src_node_embeddings = self._encode_node_sequences(
            center_node_ids=src_node_ids,
            counterpart_node_ids=dst_node_ids,
            node_interact_times=node_interact_times,
        )
        dst_node_embeddings = self._encode_node_sequences(
            center_node_ids=dst_node_ids,
            counterpart_node_ids=src_node_ids,
            node_interact_times=node_interact_times,
        )
        return src_node_embeddings, dst_node_embeddings

    def _encode_node_sequences(
        self,
        center_node_ids: np.ndarray,
        counterpart_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
    ) -> torch.Tensor:
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            center_node_ids, node_interact_times, self.num_neighbors
        )

        seq_node_ids = np.concatenate((neighbor_node_ids, counterpart_node_ids[:, np.newaxis]), axis=1)
        seq_times = np.concatenate((neighbor_times, node_interact_times[:, np.newaxis]), axis=1)
        padding_mask = np.concatenate(
            (neighbor_times <= 0.0, np.zeros((len(center_node_ids), 1), dtype=bool)),
            axis=1,
        )

        device = self.node_raw_features.device
        seq_node_ids_tensor = torch.from_numpy(seq_node_ids).long().to(device)
        neighbor_edge_ids_tensor = torch.from_numpy(neighbor_edge_ids).long().to(device)
        padding_mask_tensor = torch.from_numpy(padding_mask).bool().to(device)

        node_features = self.node_raw_features[seq_node_ids_tensor]
        neighbor_edge_features = self.edge_raw_features[neighbor_edge_ids_tensor]
        current_edge_padding = torch.zeros(
            (neighbor_edge_features.shape[0], 1, neighbor_edge_features.shape[2]),
            dtype=neighbor_edge_features.dtype,
            device=device,
        )
        edge_features = torch.cat((neighbor_edge_features, current_edge_padding), dim=1)

        seq_len = seq_node_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(seq_node_ids.shape[0], -1)
        position_features = self.position_embedding(position_ids)

        time_deltas = np.maximum(node_interact_times[:, np.newaxis] - seq_times, 0.0)
        time_deltas_tensor = torch.from_numpy(time_deltas).float().to(device)
        time_features = self.time_projection(self.time_encoder(time_deltas_tensor))

        sequence_features = (
            self.node_projection(node_features)
            + self.edge_projection(edge_features)
            + time_features
            + position_features
        )
        sequence_features = self.input_norm(self.input_dropout(sequence_features))

        encoded_sequences = self.sequence_encoder(
            sequence_features, src_key_padding_mask=padding_mask_tensor
        )
        encoded_sequences = self.output_norm(encoded_sequences + self.output_mlp(encoded_sequences))

        last_token = encoded_sequences[:, -1, :]
        valid_mask = (~padding_mask_tensor).float().unsqueeze(-1)
        pooled_tokens = torch.sum(encoded_sequences * valid_mask, dim=1) / torch.clamp(
            valid_mask.sum(dim=1), min=1.0
        )
        return self.readout(torch.cat((last_token, pooled_tokens), dim=1))
