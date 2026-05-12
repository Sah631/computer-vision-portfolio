import argparse
from .utils import train_loop
from .dataset import get_dataloaders
import torch
from .modules import ResNet18_CIFAR10
from .utils import plot_metrics
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from common.checkpoint import save_checkpoint
from common.reproducibility import set_seed

SCRIPT_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--iterations", type=int, default=64000, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Initial learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on (cuda or cpu)")
    parser.add_argument("--plot_figures", action="store_true", help="Whether to plot training metrics after training")
    parser.add_argument("--save_checkpoint", action="store_true", help="Whether to save the final model checkpoint after training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    set_seed(args.seed)

    trainloader, testloader = get_dataloaders(batch_size=args.batch_size)

    model = ResNet18_CIFAR10(in_channels=3).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimiser, milestones=[args.iterations // 2, (args.iterations * 3) // 4], gamma=0.1)

    print(f"Training with {args.iterations} iterations, batch size {args.batch_size}, and learning rate {args.learning_rate}")

    train_losses, train_accuracies, test_accuracies, loss_iterations, accuracy_iterations = train_loop( 
        dataloader=trainloader,
        model=model,
        criterion=criterion,
        optimiser=optimiser,
        iterations=args.iterations,
        device=args.device,
        scheduler=scheduler,
        trainloader=trainloader,
        testloader=testloader
    )

    if args.plot_figures:
        folder_path = SCRIPT_DIR / "figures"
        folder_path.mkdir(exist_ok=True) # Create the folder if it doesn't exist
        plot_metrics(
            train_losses,
            train_accuracies,
            test_accuracies,
            loss_iterations,
            accuracy_iterations,
            output_dir=folder_path
        )
        print(f"Training metrics saved to {folder_path.absolute()}")

    if args.save_checkpoint:
        checkpoint_path = save_checkpoint(
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            output_dir=SCRIPT_DIR / "checkpoints",
            checkpoint_name="resnet18_cifar10_final.pt",
            metadata={
                "iterations": args.iterations,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            }
        )
        print(f"Checkpoint saved to {checkpoint_path.absolute()}")
