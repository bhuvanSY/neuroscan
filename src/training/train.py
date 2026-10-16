"""Unified scratch training CLI for NeuroScan AI models."""

import argparse
import sys
from src.utils.config import load_config
from src.training.engine import ScratchTrainer
from src.utils.exceptions import DatasetNotFoundError


def main():
    parser = argparse.ArgumentParser(description="NeuroScan AI: Scratch Model Trainer")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "neuro_convnet", "resnet50", "densenet121", "efficientnet_v2", "all"],
        help="Model architecture to train from scratch",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--patience", "--early-stopping", dest="patience", type=int, default=None, help="Early stopping patience epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    args = parser.parse_args()

    # Model config mappings
    config_map = {
        "custom_cnn": "configs/custom_cnn_config.yaml",
        "neuro_convnet": "configs/custom_cnn_config.yaml",
        "resnet50": "configs/resnet50_config.yaml",
        "densenet121": "configs/densenet121_config.yaml",
        "efficientnet_v2": "configs/efficientnet_v2_config.yaml",
    }

    models_to_train = ["custom_cnn", "resnet50", "densenet121", "efficientnet_v2"] if args.model == "all" else [args.model]

    for m in models_to_train:
        cfg_file = args.config or config_map.get(m, "configs/dataset_config.yaml")
        config = load_config(cfg_file)

        if args.epochs:
            config.setdefault("training", {})["epochs"] = args.epochs
        if args.patience is not None:
            config.setdefault("training", {})["early_stopping_patience"] = args.patience
        if args.batch_size:
            config.setdefault("training", {})["batch_size"] = args.batch_size
        if args.lr:
            config.setdefault("training", {})["learning_rate"] = args.lr
        if args.device:
            config.setdefault("hardware", {})["device"] = args.device

        try:
            canonical_name = "custom_cnn" if m in ["custom_cnn", "neuro_convnet"] else m
            trainer = ScratchTrainer(model_name=canonical_name, config=config, device_preference=args.device)
            res = trainer.run()
            print(f"\n[DONE] {m.upper()} trained successfully. Checkpoint: {res['checkpoint_path']}\n")
        except DatasetNotFoundError as e:
            print(f"\n[ERROR] {e.message}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
