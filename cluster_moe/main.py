"""
Cluster of Experts (CoE) Model Architecture

A novel neural network architecture that extends Mixture of Experts (MoE) with
hierarchical expert clustering and dynamic tree-based routing for improved
scalability and specialization.

Author: Claude AI
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from loguru import logger
from dataclasses import dataclass


@dataclass
class CoEConfig:
    """Configuration class for Cluster of Experts model."""

    # Model dimensions
    d_model: int = 512
    d_ff: int = 2048

    # Expert configuration
    num_clusters: int = 4
    experts_per_cluster: int = 8
    expert_capacity_factor: float = 1.25

    # Routing configuration
    top_k_clusters: int = 2
    top_k_experts: int = 2
    routing_temperature: float = 1.0

    # Tree search configuration
    max_tree_depth: int = 3
    search_beam_width: int = 4

    # Training configuration
    load_balancing_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    dropout: float = 0.1

    # Advanced features
    use_adaptive_routing: bool = True
    use_expert_selection_cache: bool = True
    enable_expert_pruning: bool = False


class Expert(nn.Module):
    """Enhanced individual expert network with robust architecture."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        expert_id: int = 0,
        activation: str = "swiglu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.expert_id = expert_id
        self.activation_type = activation

        # Input normalization
        self.input_norm = nn.LayerNorm(d_model)

        # Main feed-forward network with residual connection
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # For SwiGLU
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation functions
        if activation == "swiglu":
            self.activation = self._swiglu
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Expert specialization and reliability metrics
        self.register_buffer(
            "usage_count", torch.tensor(0, dtype=torch.long)
        )
        self.register_buffer("total_load", torch.tensor(0.0))
        self.register_buffer("expert_loss", torch.tensor(0.0))
        self.register_buffer("reliability_score", torch.tensor(1.0))

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize expert weights with proper scaling."""
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.w1.weight, gain=0.02)
        nn.init.xavier_uniform_(self.w2.weight, gain=0.02)
        nn.init.xavier_uniform_(self.w3.weight, gain=0.02)

    def _swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation function."""
        return F.silu(x) * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass through expert with residual connections.

        Args:
            x: Input tensor [batch_size, seq_len, d_model] or [batch_size, d_model]

        Returns:
            Processed tensor with same shape as input
        """
        # Track usage
        self.usage_count += 1
        self.total_load += (
            x.numel() / self.d_model
        )  # Normalize by sequence length

        # Input normalization
        x_norm = self.input_norm(x)

        # Main computation with residual connection
        if self.activation_type == "swiglu":
            # SwiGLU: (xW1 * SiLU(xW2))W3
            h1 = self.w1(x_norm)
            h2 = self.w2(x_norm)
            h = self.activation(h2) * h1
        else:
            # Standard activation: activation(xW1)W3
            h = self.activation(self.w1(x_norm))

        h = self.dropout1(h)
        h = self.w3(h)
        h = self.dropout2(h)

        # Residual connection and output normalization
        output = x + h  # Residual connection
        output = self.output_norm(output)

        return output

    def get_specialization_score(self) -> float:
        """Calculate expert specialization score based on usage patterns."""
        if self.usage_count.item() == 0:
            return 0.0
        return self.total_load.item() / self.usage_count.item()

    def get_reliability_score(self) -> float:
        """Get expert reliability score."""
        return self.reliability_score.item()

    def update_reliability(self, loss: float):
        """Update expert reliability based on performance."""
        # Exponential moving average of loss
        alpha = 0.9
        self.expert_loss = (
            alpha * self.expert_loss + (1 - alpha) * loss
        )
        # Reliability decreases with higher loss
        self.reliability_score = torch.exp(-self.expert_loss)

    def reset_metrics(self):
        """Reset expert metrics for new training phase."""
        self.usage_count.zero_()
        self.total_load.zero_()
        self.expert_loss.zero_()
        self.reliability_score.fill_(1.0)


class TreeRouter(nn.Module):
    """Enhanced hierarchical tree-based router with reliability-aware routing."""

    def __init__(
        self,
        d_model: int,
        num_clusters: int,
        experts_per_cluster: int,
        max_depth: int = 3,
        beam_width: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.max_depth = max_depth
        self.beam_width = beam_width

        # Enhanced cluster-level routing with attention
        self.cluster_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_clusters),
        )

        # Enhanced expert-level routing within each cluster
        self.expert_routers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.LayerNorm(d_model // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model // 2, experts_per_cluster),
                )
                for _ in range(num_clusters)
            ]
        )

        # Adaptive routing weights with reliability awareness
        self.routing_adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Reliability tracking
        self.register_buffer(
            "cluster_reliabilities", torch.ones(num_clusters)
        )
        self.register_buffer(
            "expert_reliabilities",
            torch.ones(num_clusters, experts_per_cluster),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize router weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def update_reliabilities(
        self,
        cluster_reliabilities: torch.Tensor,
        expert_reliabilities: torch.Tensor,
    ):
        """Update reliability scores for routing decisions."""
        alpha = 0.9
        self.cluster_reliabilities = (
            alpha * self.cluster_reliabilities
            + (1 - alpha) * cluster_reliabilities
        )
        self.expert_reliabilities = (
            alpha * self.expert_reliabilities
            + (1 - alpha) * expert_reliabilities
        )

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        top_k_clusters: int = 2,
        top_k_experts: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced hierarchical routing through the expert tree.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            temperature: Routing temperature for softmax
            top_k_clusters: Number of top clusters to select
            top_k_experts: Number of top experts per cluster to select

        Returns:
            Dict containing routing weights and selections
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten for routing
        x_flat = x.view(-1, d_model)

        # Step 1: Enhanced cluster-level routing with reliability weighting
        cluster_logits = self.cluster_router(x_flat)

        # Apply reliability weighting
        reliability_weights = self.cluster_reliabilities.unsqueeze(
            0
        ).expand_as(cluster_logits)
        cluster_logits = cluster_logits * reliability_weights

        cluster_logits = cluster_logits / temperature
        cluster_probs = F.softmax(cluster_logits, dim=-1)

        # Select top-k clusters with noise for exploration
        if self.training:
            # Add noise during training for better exploration
            noise = torch.randn_like(cluster_probs) * 0.1
            cluster_probs = F.softmax(cluster_logits + noise, dim=-1)

        cluster_weights, cluster_indices = torch.topk(
            cluster_probs, top_k_clusters, dim=-1
        )

        # Step 2: Enhanced expert-level routing within selected clusters
        expert_weights_list = []
        expert_indices_list = []

        for i in range(top_k_clusters):
            cluster_idx = cluster_indices[:, i]

            # Get expert logits for this cluster
            expert_logits = torch.zeros(
                x_flat.size(0),
                self.experts_per_cluster,
                device=x.device,
                dtype=x.dtype,
            )

            for j in range(self.num_clusters):
                mask = cluster_idx == j
                if mask.any():
                    expert_logits[mask] = self.expert_routers[j](
                        x_flat[mask]
                    )

                    # Apply expert reliability weighting
                    expert_reliability = self.expert_reliabilities[
                        j
                    ].unsqueeze(0)
                    expert_logits[mask] = (
                        expert_logits[mask] * expert_reliability
                    )

            expert_logits = expert_logits / temperature
            expert_probs = F.softmax(expert_logits, dim=-1)

            # Select top-k experts within cluster
            expert_weights, expert_indices = torch.topk(
                expert_probs, top_k_experts, dim=-1
            )

            expert_weights_list.append(expert_weights)
            expert_indices_list.append(expert_indices)

        # Step 3: Enhanced adaptive routing weight calculation
        adaptive_weights = self.routing_adapter(x_flat)

        return {
            "cluster_weights": cluster_weights,
            "cluster_indices": cluster_indices,
            "expert_weights": expert_weights_list,
            "expert_indices": expert_indices_list,
            "adaptive_weights": adaptive_weights.view(
                batch_size, seq_len, 1
            ),
        }


class ExpertCluster(nn.Module):
    """Enhanced cluster of experts with advanced load balancing and reliability tracking."""

    def __init__(
        self,
        cluster_id: int,
        d_model: int,
        d_ff: int,
        num_experts: int,
        dropout: float = 0.1,
        activation: str = "swiglu",
    ):
        super().__init__()
        self.cluster_id = cluster_id
        self.num_experts = num_experts
        self.d_model = d_model

        # Create enhanced experts with different activations for diversity
        activations = (
            ["swiglu", "gelu", "relu"]
            if num_experts >= 3
            else ["swiglu"]
        )
        self.experts = nn.ModuleList(
            [
                Expert(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    expert_id=i,
                    activation=activations[i % len(activations)],
                )
                for i in range(num_experts)
            ]
        )

        # Cluster-specific normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Enhanced load balancing with diversity tracking
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer(
            "expert_diversity", torch.ones(num_experts)
        )
        self.register_buffer("cluster_reliability", torch.tensor(1.0))

        # Expert capacity management
        self.expert_capacity = torch.ones(num_experts)
        self.register_buffer(
            "capacity_usage", torch.zeros(num_experts)
        )

        # Initialize cluster
        self._initialize_cluster()

    def _initialize_cluster(self):
        """Initialize cluster with proper weight scaling."""
        # Scale expert capacities based on model size
        total_params = sum(
            p.numel()
            for expert in self.experts
            for p in expert.parameters()
        )
        self.expert_capacity = torch.ones(self.num_experts) * (
            total_params / self.num_experts
        )

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enhanced forward pass through selected experts in the cluster.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            expert_weights: Weights for selected experts [batch_size, seq_len, top_k_experts]
            expert_indices: Indices of selected experts [batch_size, seq_len, top_k_experts]

        Returns:
            Weighted combination of expert outputs [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(
            -1, d_model
        )  # [batch_size * seq_len, d_model]

        # Reshape routing info
        expert_weights_flat = expert_weights.view(
            -1, expert_weights.size(-1)
        )
        expert_indices_flat = expert_indices.view(
            -1, expert_indices.size(-1)
        )

        # Apply capacity constraints and reliability weighting
        expert_weights_flat = self._apply_capacity_constraints(
            expert_weights_flat, expert_indices_flat
        )

        # Normalize weights
        expert_weights_flat = F.softmax(expert_weights_flat, dim=-1)

        # Compute expert outputs with reliability tracking
        expert_outputs = []
        for i in range(expert_weights_flat.size(1)):  # top_k_experts
            expert_idx = expert_indices_flat[:, i]
            weights = expert_weights_flat[:, i].unsqueeze(-1)

            # Process through selected expert
            output = torch.zeros_like(x_flat)
            for j in range(self.num_experts):
                mask = expert_idx == j
                if mask.any():
                    # Get expert output
                    expert_output = self.experts[j](x_flat[mask])

                    # Apply reliability weighting
                    reliability_weight = self.experts[
                        j
                    ].get_reliability_score()
                    expert_output = expert_output * reliability_weight

                    output[mask] = expert_output

                    # Update usage tracking
                    self.expert_usage[j] += mask.sum().item()
                    self.capacity_usage[j] += mask.sum().item()

            expert_outputs.append(weights * output)

        # Combine expert outputs
        combined_output = sum(expert_outputs)
        combined_output = combined_output.view(
            batch_size, seq_len, d_model
        )

        # Apply cluster normalization
        output = self.layer_norm(combined_output)

        # Update cluster reliability
        self._update_cluster_reliability()

        return output

    def _apply_capacity_constraints(
        self,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply capacity constraints to expert weights."""
        # Calculate capacity utilization
        capacity_utilization = self.capacity_usage / (
            self.expert_capacity + 1e-8
        )

        # Penalize over-utilized experts
        for i in range(self.num_experts):
            mask = expert_indices == i
            if mask.any():
                utilization_penalty = torch.clamp(
                    capacity_utilization[i], 0.0, 2.0
                )
                expert_weights[mask] *= 1.0 / (
                    1.0 + utilization_penalty
                )

        return expert_weights

    def _update_cluster_reliability(self):
        """Update cluster reliability based on expert performance."""
        # Average reliability of all experts
        expert_reliabilities = torch.tensor(
            [
                expert.get_reliability_score()
                for expert in self.experts
            ]
        )

        # Weight by usage
        usage_weights = self.expert_usage / (
            self.expert_usage.sum() + 1e-8
        )
        weighted_reliability = (
            expert_reliabilities * usage_weights
        ).sum()

        # Update cluster reliability
        alpha = 0.95
        self.cluster_reliability = (
            alpha * self.cluster_reliability
            + (1 - alpha) * weighted_reliability
        )

    def get_load_balancing_loss(self) -> torch.Tensor:
        """Calculate enhanced load balancing loss."""
        # Standard load balancing
        usage_probs = self.expert_usage / (
            self.expert_usage.sum() + 1e-8
        )
        uniform_prob = 1.0 / self.num_experts
        lb_loss = F.kl_div(
            usage_probs.log(),
            torch.full_like(usage_probs, uniform_prob),
            reduction="batchmean",
        )

        # Capacity balancing loss
        capacity_utilization = self.capacity_usage / (
            self.expert_capacity + 1e-8
        )
        capacity_variance = torch.var(capacity_utilization)
        capacity_loss = capacity_variance * 0.1

        return lb_loss + capacity_loss

    def get_diversity_loss(self) -> torch.Tensor:
        """Calculate expert diversity loss to encourage specialization."""
        # Calculate usage probabilities
        usage_probs = self.expert_usage / (
            self.expert_usage.sum() + 1e-8
        )

        # Encourage different experts to be used
        usage_entropy = -torch.sum(
            usage_probs * torch.log(usage_probs + 1e-8)
        )
        target_entropy = torch.log(
            torch.tensor(float(self.num_experts))
        )
        diversity_loss = F.mse_loss(usage_entropy, target_entropy)

        return diversity_loss

    def get_cluster_reliability(self) -> float:
        """Get cluster reliability score."""
        return self.cluster_reliability.item()

    def reset_metrics(self):
        """Reset cluster metrics."""
        self.expert_usage.zero_()
        self.capacity_usage.zero_()
        self.cluster_reliability.fill_(1.0)

        # Reset expert metrics
        for expert in self.experts:
            expert.reset_metrics()


class ClusterOfExpertsLayer(nn.Module):
    """Enhanced Cluster of Experts layer with improved reliability and stability."""

    def __init__(self, config: CoEConfig):
        super().__init__()
        self.config = config

        logger.info(
            f"Initializing enhanced CoE layer with {config.num_clusters} clusters, "
            f"{config.experts_per_cluster} experts per cluster"
        )

        # Create enhanced expert clusters
        self.clusters = nn.ModuleList(
            [
                ExpertCluster(
                    cluster_id=i,
                    d_model=config.d_model,
                    d_ff=config.d_ff,
                    num_experts=config.experts_per_cluster,
                    dropout=config.dropout,
                    activation="swiglu",
                )
                for i in range(config.num_clusters)
            ]
        )

        # Enhanced tree-based router
        self.router = TreeRouter(
            d_model=config.d_model,
            num_clusters=config.num_clusters,
            experts_per_cluster=config.experts_per_cluster,
            max_depth=config.max_tree_depth,
            beam_width=config.search_beam_width,
        )

        # Enhanced input/output projections
        self.input_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

        # Expert selection cache
        if config.use_expert_selection_cache:
            self.selection_cache = {}

        # Layer reliability tracking
        self.register_buffer("layer_reliability", torch.tensor(1.0))
        self.register_buffer("routing_entropy", torch.tensor(0.0))

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Enhanced forward pass through the Cluster of Experts layer.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            training: Whether in training mode

        Returns:
            Tuple of (output_tensor, routing_info)
        """
        residual = x
        x = self.input_norm(x)

        # Perform enhanced hierarchical routing
        routing_info = self.router(
            x,
            temperature=self.config.routing_temperature,
            top_k_clusters=self.config.top_k_clusters,
            top_k_experts=self.config.top_k_experts,
        )

        # Process through selected clusters
        cluster_outputs = []
        batch_size, seq_len, d_model = x.shape

        # Reshape routing info to match input dimensions
        cluster_weights = routing_info["cluster_weights"].view(
            batch_size, seq_len, -1
        )
        cluster_indices = routing_info["cluster_indices"].view(
            batch_size, seq_len, -1
        )

        for i in range(self.config.top_k_clusters):
            cluster_idx = cluster_indices[
                :, :, i
            ]  # [batch_size, seq_len]
            cluster_weight = cluster_weights[:, :, i].unsqueeze(
                -1
            )  # [batch_size, seq_len, 1]

            # Get expert routing info for this cluster
            expert_weights = routing_info["expert_weights"][i].view(
                batch_size, seq_len, -1
            )
            expert_indices = routing_info["expert_indices"][i].view(
                batch_size, seq_len, -1
            )

            # Process through cluster
            cluster_output = torch.zeros_like(x)

            for j in range(self.config.num_clusters):
                # Create mask for this cluster
                mask = cluster_idx == j  # [batch_size, seq_len]

                if mask.any():
                    # Get masked inputs and routing info
                    masked_x = x[mask]  # [num_masked_tokens, d_model]
                    masked_expert_weights = expert_weights[
                        mask
                    ]  # [num_masked_tokens, top_k_experts]
                    masked_expert_indices = expert_indices[
                        mask
                    ]  # [num_masked_tokens, top_k_experts]

                    # Process through cluster
                    cluster_out = self.clusters[j](
                        masked_x.unsqueeze(0),  # Add batch dimension
                        masked_expert_weights.unsqueeze(0),
                        masked_expert_indices.unsqueeze(0),
                    )

                    # Remove batch dimension and assign back
                    cluster_output[mask] = cluster_out.squeeze(0)

            cluster_outputs.append(cluster_weight * cluster_output)

        # Combine cluster outputs
        output = sum(cluster_outputs)

        # Apply enhanced adaptive weighting
        if self.config.use_adaptive_routing:
            adaptive_weights = routing_info["adaptive_weights"]
            output = output * adaptive_weights

        # Enhanced output projection and residual connection
        output = self.output_projection(output)
        output = output + residual

        # Update router reliabilities
        self._update_router_reliabilities()

        # Prepare enhanced auxiliary losses
        aux_losses = self._compute_enhanced_auxiliary_losses(
            routing_info
        )

        return output, {
            "routing_info": routing_info,
            "aux_losses": aux_losses,
        }

    def _update_router_reliabilities(self):
        """Update router reliability scores based on cluster performance."""
        cluster_reliabilities = torch.tensor(
            [
                cluster.get_cluster_reliability()
                for cluster in self.clusters
            ]
        )

        expert_reliabilities = torch.stack(
            [
                torch.tensor(
                    [
                        expert.get_reliability_score()
                        for expert in cluster.experts
                    ]
                )
                for cluster in self.clusters
            ]
        )

        self.router.update_reliabilities(
            cluster_reliabilities, expert_reliabilities
        )

    def _compute_enhanced_auxiliary_losses(
        self, routing_info: Dict
    ) -> Dict[str, torch.Tensor]:
        """Compute enhanced auxiliary losses for training."""
        aux_losses = {}

        # Enhanced load balancing loss
        if self.config.load_balancing_weight > 0:
            lb_loss = sum(
                cluster.get_load_balancing_loss()
                for cluster in self.clusters
            )
            aux_losses["load_balancing"] = (
                self.config.load_balancing_weight * lb_loss
            )

        # Enhanced router z-loss
        if self.config.router_z_loss_weight > 0:
            cluster_logits = routing_info["cluster_weights"]
            z_loss = torch.mean(
                torch.square(torch.logsumexp(cluster_logits, dim=-1))
            )
            aux_losses["router_z_loss"] = (
                self.config.router_z_loss_weight * z_loss
            )

        # Diversity loss to encourage expert specialization
        diversity_loss = sum(
            cluster.get_diversity_loss() for cluster in self.clusters
        )
        aux_losses["diversity"] = 0.01 * diversity_loss

        # Routing entropy loss for better exploration
        cluster_probs = F.softmax(
            routing_info["cluster_weights"], dim=-1
        )
        routing_entropy = -torch.mean(
            torch.sum(
                cluster_probs * torch.log(cluster_probs + 1e-8),
                dim=-1,
            )
        )
        target_entropy = torch.log(
            torch.tensor(float(self.config.num_clusters))
        )
        entropy_loss = F.mse_loss(routing_entropy, target_entropy)
        aux_losses["routing_entropy"] = 0.005 * entropy_loss

        return aux_losses

    def get_layer_reliability(self) -> float:
        """Get layer reliability score."""
        return self.layer_reliability.item()

    def reset_metrics(self):
        """Reset layer metrics."""
        self.layer_reliability.fill_(1.0)
        self.routing_entropy.zero_()

        # Reset cluster metrics
        for cluster in self.clusters:
            cluster.reset_metrics()


class ClusterOfExpertsModel(nn.Module):
    """Enhanced Cluster of Experts model with improved reliability and monitoring."""

    def __init__(
        self,
        config: CoEConfig,
        num_layers: int = 6,
        vocab_size: int = 32000,
    ):
        super().__init__()
        self.config = config
        self.num_layers = num_layers

        logger.info(
            f"Building enhanced CoE model with {num_layers} layers"
        )

        # Enhanced token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size, config.d_model
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1024, config.d_model) * 0.02
        )

        # Enhanced CoE layers
        self.coe_layers = nn.ModuleList(
            [ClusterOfExpertsLayer(config) for _ in range(num_layers)]
        )

        # Enhanced output head
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(
            config.d_model, vocab_size, bias=False
        )

        # Model reliability tracking
        self.register_buffer("model_reliability", torch.tensor(1.0))
        self.register_buffer(
            "training_step", torch.tensor(0, dtype=torch.long)
        )

        # Initialize weights
        self._initialize_weights()

        # Model statistics
        self.total_params = sum(p.numel() for p in self.parameters())
        self.expert_params = sum(
            p.numel()
            for layer in self.coe_layers
            for cluster in layer.clusters
            for expert in cluster.experts
            for p in expert.parameters()
        )

        logger.info(
            f"Enhanced model initialized: {self.total_params:,} total params, "
            f"{self.expert_params:,} expert params"
        )

    def _initialize_weights(self):
        """Initialize model weights with improved scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(
                    module.weight, gain=0.02
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass through the complete model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Dict containing logits and enhanced auxiliary information
        """
        batch_size, seq_len = input_ids.shape

        # Token and position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding[:seq_len].unsqueeze(0)
        x = token_emb + pos_emb

        # Process through enhanced CoE layers
        all_aux_losses = []
        routing_stats = []
        layer_reliabilities = []

        for i, layer in enumerate(self.coe_layers):
            x, layer_info = layer(x, training=self.training)
            all_aux_losses.append(layer_info["aux_losses"])
            routing_stats.append(layer_info["routing_info"])
            layer_reliabilities.append(layer.get_layer_reliability())

        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.output_projection(x)

        # Aggregate enhanced auxiliary losses
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for aux_losses in all_aux_losses:
            for loss in aux_losses.values():
                total_aux_loss = total_aux_loss + loss

        # Update model reliability
        self._update_model_reliability(layer_reliabilities)

        # Update training step
        self.training_step += 1

        return {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "routing_stats": routing_stats,
            "model_reliability": self.model_reliability,
            "layer_reliabilities": layer_reliabilities,
        }

    def _update_model_reliability(
        self, layer_reliabilities: List[float]
    ):
        """Update model reliability based on layer performance."""
        avg_layer_reliability = torch.tensor(
            layer_reliabilities
        ).mean()
        alpha = 0.95
        self.model_reliability = (
            alpha * self.model_reliability
            + (1 - alpha) * avg_layer_reliability
        )

    def get_expert_utilization(self) -> Dict[str, float]:
        """Get enhanced expert utilization statistics."""
        stats = {}
        for layer_idx, layer in enumerate(self.coe_layers):
            for cluster_idx, cluster in enumerate(layer.clusters):
                for expert_idx, expert in enumerate(cluster.experts):
                    key = f"layer_{layer_idx}_cluster_{cluster_idx}_expert_{expert_idx}"
                    stats[key] = expert.get_specialization_score()
        return stats

    def get_reliability_stats(self) -> Dict[str, float]:
        """Get comprehensive reliability statistics."""
        stats = {
            "model_reliability": self.model_reliability.item(),
            "training_step": self.training_step.item(),
        }

        # Layer reliabilities
        for layer_idx, layer in enumerate(self.coe_layers):
            stats[f"layer_{layer_idx}_reliability"] = (
                layer.get_layer_reliability()
            )

            # Cluster reliabilities
            for cluster_idx, cluster in enumerate(layer.clusters):
                stats[
                    f"layer_{layer_idx}_cluster_{cluster_idx}_reliability"
                ] = cluster.get_cluster_reliability()

                # Expert reliabilities
                for expert_idx, expert in enumerate(cluster.experts):
                    stats[
                        f"layer_{layer_idx}_cluster_{cluster_idx}_expert_{expert_idx}_reliability"
                    ] = expert.get_reliability_score()

        return stats

    def reset_metrics(self):
        """Reset all model metrics."""
        self.model_reliability.fill_(1.0)
        self.training_step.zero_()

        # Reset layer metrics
        for layer in self.coe_layers:
            layer.reset_metrics()

    def update_expert_reliabilities(self, losses: Dict[str, float]):
        """Update expert reliabilities based on task-specific losses."""
        for layer_idx, layer in enumerate(self.coe_layers):
            for cluster_idx, cluster in enumerate(layer.clusters):
                for expert_idx, expert in enumerate(cluster.experts):
                    key = f"layer_{layer_idx}_cluster_{cluster_idx}_expert_{expert_idx}"
                    if key in losses:
                        expert.update_reliability(losses[key])

    def get_model_summary(self) -> Dict[str, any]:
        """Get comprehensive model summary."""
        return {
            "total_params": self.total_params,
            "expert_params": self.expert_params,
            "num_layers": self.num_layers,
            "num_clusters": self.config.num_clusters,
            "experts_per_cluster": self.config.experts_per_cluster,
            "total_experts": self.num_layers
            * self.config.num_clusters
            * self.config.experts_per_cluster,
            "model_reliability": self.model_reliability.item(),
            "training_step": self.training_step.item(),
        }


def create_coe_model(
    d_model: int = 512,
    num_layers: int = 6,
    num_clusters: int = 4,
    experts_per_cluster: int = 8,
    vocab_size: int = 32000,
    **kwargs,
) -> ClusterOfExpertsModel:
    """
    Enhanced factory function to create a Cluster of Experts model.

    Args:
        d_model: Model dimension
        num_layers: Number of CoE layers
        num_clusters: Number of expert clusters per layer
        experts_per_cluster: Number of experts per cluster
        vocab_size: Vocabulary size
        **kwargs: Additional configuration parameters

    Returns:
        Initialized ClusterOfExpertsModel
    """
    config = CoEConfig(
        d_model=d_model,
        num_clusters=num_clusters,
        experts_per_cluster=experts_per_cluster,
        **kwargs,
    )

    model = ClusterOfExpertsModel(config, num_layers, vocab_size)

    logger.info("Enhanced CoE model created successfully")
    logger.info(
        f"Architecture: {num_layers} layers, {num_clusters} clusters, "
        f"{experts_per_cluster} experts/cluster"
    )
    logger.info(
        f"Total experts: {num_layers * num_clusters * experts_per_cluster}"
    )

    # Log model summary
    summary = model.get_model_summary()
    logger.info(f"Model summary: {summary}")

    return model


# # Enhanced example usage and testing
# if __name__ == "__main__":
#     # Configure logging
#     logger.add("enhanced_coe_model.log", rotation="500 MB")
#     logger.info("Starting Enhanced Cluster of Experts model test")

#     # Create enhanced model
#     model = create_coe_model(
#         d_model=512,
#         num_layers=4,
#         num_clusters=3,
#         experts_per_cluster=4,
#         vocab_size=1000,
#     )

#     # Test forward pass
#     batch_size, seq_len = 2, 10
#     input_ids = torch.randint(0, 1000, (batch_size, seq_len))

#     logger.info(
#         f"Testing enhanced forward pass with input shape: {input_ids.shape}"
#     )

#     with torch.no_grad():
#         output = model(input_ids)
#         logger.info(f"Output logits shape: {output['logits'].shape}")
#         logger.info(
#             f"Enhanced auxiliary loss: {output['aux_loss'].item():.6f}"
#         )
#         logger.info(
#             f"Model reliability: {output['model_reliability'].item():.4f}"
#         )

#     # Enhanced expert utilization
#     utilization = model.get_expert_utilization()
#     logger.info("Enhanced expert utilization stats:")
#     for expert, score in list(utilization.items())[
#         :5
#     ]:  # Show first 5
#         logger.info(f"  {expert}: {score:.4f}")

#     # Reliability statistics
#     reliability_stats = model.get_reliability_stats()
#     logger.info("Reliability statistics:")
#     logger.info(
#         f"  Model reliability: {reliability_stats['model_reliability']:.4f}"
#     )
#     logger.info(
#         f"  Training step: {reliability_stats['training_step']}"
#     )

#     # Model summary
#     summary = model.get_model_summary()
#     logger.info("Enhanced model summary:")
#     for key, value in summary.items():
#         logger.info(f"  {key}: {value}")

#     logger.info("Enhanced CoE model test completed successfully")
