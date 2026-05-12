from pathlib import Path

import torch


def save_checkpoint(model, optimiser, scheduler, output_dir, checkpoint_name, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / checkpoint_name

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path
