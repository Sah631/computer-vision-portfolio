import argparse
from .utils import check_accuracy
from .dataset import get_dataloaders
import torch
from pathlib import Path
from .modules import ResNet18_CIFAR10

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to test on (cuda or cpu)")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/resnet18_cifar10_final.pt", help="Path to the model checkpoint for testing")
    args = parser.parse_args()

    testloader = get_dataloaders(batch_size=args.batch_size)[1]

    checkpoint_path = SCRIPT_DIR / args.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    model = ResNet18_CIFAR10(in_channels=3).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_accuracy = check_accuracy(testloader, model, args.device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")