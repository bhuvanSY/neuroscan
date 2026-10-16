"""Independent Training Script for MODEL A: Custom CNN."""

import argparse
import sys
from src.utils.config import load_config
from src.training.engine import ScratchTrainer
from src.utils.exceptions import DatasetNotFoundError


def main():
    parser = argparse.ArgumentParser(description="NeuroScan AI - Train Custom CNN from Scratch")
    parser.add_argument("--config", type=str, default="configs/custom_cnn_config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr

    try:
        trainer = ScratchTrainer(model_name="custom_cnn", config=config)
        res = trainer.run()
        print("\n" + "="*60)
        print("MODEL A (Custom CNN) SCRATCH TRAINING COMPLETE")
        print(f"Best Validation Macro-F1: {res['best_val_macro_f1']:.4f}")
        print(f"Best Validation Accuracy: {res['best_val_accuracy']:.4f}")
        print(f"Checkpoint saved to: {res['checkpoint_path']}")
        print(f"Learning curves saved to: {res['curves_path']}")
        print("="*60 + "\n")
    except DatasetNotFoundError as e:
        print(f"\n[ERROR] {e.message}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
