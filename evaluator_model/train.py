"""
Training script for the ResponseQualityModel.
"""

import argparse
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import structlog

from evaluator_model.models import ResponseQualityModel

logger = structlog.get_logger()


# TODO: Create proper dataset classes in data.py
# For now, define a minimal placeholder
class PlaceholderDataset(Dataset):
    """
    Placeholder dataset for training.
    
    TODO: Replace with real dataset that loads:
    - prompts
    - weak_responses
    - strong_responses (optional)
    - labels (0/1 for adequacy, or preference scores)
    """
    
    def __init__(self, data_path: Path):
        """Load dataset from disk."""
        # TODO: Implement real data loading
        # Example: pd.read_csv(data_path)
        self.data = []
        logger.warning("using_placeholder_dataset")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        # TODO: Return real data
        return {
            "prompt": "What is Python?",
            "response": "Python is a programming language.",
            "label": 1.0,
        }


def train_epoch(
    model: ResponseQualityModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[any] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: ResponseQualityModel instance
        dataloader: Training DataLoader
        criterion: Loss function (e.g., nn.BCELoss)
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Optional learning rate scheduler
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Dictionary with training metrics (loss, accuracy, etc.)
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # TODO: Adjust based on your actual dataset format
        # Assuming batch contains: prompt, response, label
        prompts = batch["prompt"]
        responses = batch["response"]
        labels = batch["label"].to(device)  # Should be shape [batch_size, 1]
        
        # Encode each pair in the batch
        # TODO: Optimize this with proper batching
        input_ids_list = []
        attention_mask_list = []
        
        for prompt, response in zip(prompts, responses):
            encoding = model.encode_pair(prompt, response, device=device)
            input_ids_list.append(encoding["input_ids"])
            attention_mask_list.append(encoding["attention_mask"])
        
        # Stack into batch tensors
        input_ids = torch.cat(input_ids_list, dim=0)  # [batch_size, seq_len]
        attention_mask = torch.cat(attention_mask_list, dim=0)
        
        # Forward pass
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)  # [batch_size, 1]
        
        # Compute loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Calculate accuracy (for binary classification)
        binary_preds = (predictions > 0.5).float()
        correct_predictions += (binary_preds == labels).sum().item()
        total_predictions += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": loss.item(),
            "acc": correct_predictions / total_predictions,
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


def evaluate(
    model: ResponseQualityModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict[str, float]:
    """
    Evaluate the model on validation set.
    
    Args:
        model: QualityEvaluator model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to evaluate on
        task_type: 'adequacy' or 'pairwise'
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if task_type == "adequacy":
                encoded = model.encode_pair(batch["prompts"], batch["responses"])
    Evaluate the model on validation set.
    
    Args:
        model: ResponseQualityModel instance
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # TODO: Adjust based on your dataset format
            prompts = batch["prompt"]
            responses = batch["response"]
            labels = batch["label"].to(device)
            
            # Encode batch
            input_ids_list = []
            attention_mask_list = []
            
            for prompt, response in zip(prompts, responses):
                encoding = model.encode_pair(prompt, response, device=device)
                input_ids_list.append(encoding["input_ids"])
                attention_mask_list.append(encoding["attention_mask"])
            
            input_ids = torch.cat(input_ids_list, dim=0)
            attention_mask = torch.cat(attention_mask_list, dim=0)
            
            # Forward pass
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            binary_preds = (predictions > 0.5).float()
            correct_predictions += (binary_preds == labels).sum().item()
            total_predictions += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


def main() -> None:
    """
    Main training function with checkpoint saving.
    
    TODO: Customize this function for your specific needs:
    - Adjust hyperparameters
    - Add early stopping
    - Implement learning rate scheduling strategies
    - Add W&B or TensorBoard logging
    """
    parser = argparse.ArgumentParser(description="Train ResponseQualityModel")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation data")
    parser.add_argument("--encoder", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("using_device", device=str(device))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("training_started", args=vars(args))
    
    # TODO: Load real dataset
    # For now, use placeholder
    logger.warning("using_placeholder_dataset", message="Replace with real dataset loading")
    train_dataset = PlaceholderDataset(Path(args.train_data))
    val_dataset = PlaceholderDataset(Path(args.val_data)) if args.val_data else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # TODO: Increase for real training
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    
    # Initialize model
    logger.info("initializing_model", encoder=args.encoder)
    model = ResponseQualityModel(encoder_name=args.encoder)
    model.to(device)
    
    # Loss function (Binary Cross Entropy for quality prediction)
    criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop with checkpoint saving
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        logger.info("epoch_start", epoch=epoch + 1, total_epochs=args.epochs)
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=args.max_grad_norm,
        )
        
        logger.info("epoch_train_complete", epoch=epoch + 1, **train_metrics)
        
        # Validate
        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )
            
            logger.info("epoch_val_complete", epoch=epoch + 1, **val_metrics)
            
            # Save best checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint_path = output_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }, checkpoint_path)
                logger.info("checkpoint_saved", path=str(checkpoint_path), val_loss=best_val_loss)
        
        # Save latest checkpoint
        latest_checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, latest_checkpoint_path)
    
    logger.info("training_complete", best_val_loss=best_val_loss)


if __name__ == "__main__":
    main()

        collate_fn = collate_pairwise
        criterion = PairwisePreferenceLoss()
    
    # Create split if no validation data provided
    if args.val_data is None:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset
        if args.task == "adequacy":
            val_dataset = AdequacyDataset.from_csv(args.val_data)
        else:
            val_dataset = PairwisePreferenceDataset.from_csv(args.val_data)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Initialize model
    model = QualityEvaluator(encoder_name=args.encoder).to(args.device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        logger.info("epoch_start", epoch=epoch + 1)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, args.device, args.task
        )
        logger.info("train_complete", epoch=epoch + 1, **train_metrics)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, args.device, args.task)
        logger.info("val_complete", epoch=epoch + 1, **val_metrics)
        
        # Save checkpoint if validation improves
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = output_dir / "evaluator_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("checkpoint_saved", path=str(checkpoint_path))
    
    logger.info("training_complete")


if __name__ == "__main__":
    main()

    
    # Split train/val
    if args.val_data:
        train_dataset = dataset
        if args.task == "adequacy":
            val_dataset = AdequacyDataset.from_csv(args.val_data)
        else:
            val_dataset = PairwisePreferenceDataset.from_csv(args.val_data)
    else:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Initialize model
    model = QualityEvaluator(encoder_name=args.encoder)
    model = model.to(args.device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        logger.info("epoch_started", epoch=epoch + 1, total=args.epochs)
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            task_type=args.task,
        )
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=args.device,
            task_type=args.task,
        )
        
        logger.info(
            "epoch_completed",
            epoch=epoch + 1,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            val_accuracy=val_metrics["accuracy"],
        )
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = output_dir / "evaluator_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "config": {
                        "encoder_name": args.encoder,
                        "task": args.task,
                    },
                },
                checkpoint_path,
            )
            logger.info("checkpoint_saved", path=str(checkpoint_path))
    
    logger.info("training_completed", best_val_loss=best_val_loss)


if __name__ == "__main__":
    main()
