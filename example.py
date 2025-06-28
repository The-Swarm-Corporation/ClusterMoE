from loguru import logger
import torch
from clustermoe.main import create_coe_model

# Enhanced example usage and testing
if __name__ == "__main__":
    # Configure logging
    logger.add("enhanced_coe_model.log", rotation="500 MB")
    logger.info("Starting Enhanced Cluster of Experts model test")

    # Create enhanced model
    model = create_coe_model(
        d_model=512,
        num_layers=4,
        num_clusters=3,
        experts_per_cluster=4,
        vocab_size=1000,
    )

    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    logger.info(
        f"Testing enhanced forward pass with input shape: {input_ids.shape}"
    )

    with torch.no_grad():
        output = model(input_ids)
        logger.info(f"Output logits shape: {output['logits'].shape}")
        logger.info(
            f"Enhanced auxiliary loss: {output['aux_loss'].item():.6f}"
        )
        logger.info(
            f"Model reliability: {output['model_reliability'].item():.4f}"
        )

    # Enhanced expert utilization
    utilization = model.get_expert_utilization()
    logger.info("Enhanced expert utilization stats:")
    for expert, score in list(utilization.items())[
        :5
    ]:  # Show first 5
        logger.info(f"  {expert}: {score:.4f}")

    # Reliability statistics
    reliability_stats = model.get_reliability_stats()
    logger.info("Reliability statistics:")
    logger.info(
        f"  Model reliability: {reliability_stats['model_reliability']:.4f}"
    )
    logger.info(
        f"  Training step: {reliability_stats['training_step']}"
    )

    # Model summary
    summary = model.get_model_summary()
    logger.info("Enhanced model summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    logger.info("Enhanced CoE model test completed successfully")
