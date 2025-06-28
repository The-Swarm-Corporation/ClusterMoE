from cluster_moe.main import (
    # Configuration
    CoEConfig,
    
    # Core components
    Expert,
    ExpertCluster,
    TreeRouter,
    ClusterOfExpertsLayer,
    ClusterOfExpertsModel,
    
    # Factory function
    create_coe_model,
)

__version__ = "1.0.0"
__author__ = "The Swarm Corporation"
__license__ = "MIT"

__all__ = [
    "CoEConfig",
    "Expert", 
    "ExpertCluster",
    "TreeRouter",
    "ClusterOfExpertsLayer",
    "ClusterOfExpertsModel",
    "create_coe_model",
]
