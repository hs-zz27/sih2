import os
import sys
import time
import argparse
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.common.checkpointing import safe_torch_load

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_latest_checkpoint(ckpt_dir):
    checkpoints = glob.glob(os.path.join(ckpt_dir, "ckpt_step_*.pt"))
    if not checkpoints:
        return None
    def extract_step(path):
        match = re.search(r"ckpt_step_(\d+)\.pt", path)
        return int(match.group(1)) if match else -1
    latest_ckpt = max(checkpoints, key=extract_step)
    return latest_ckpt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    start_step = 0
    set_seed(42)

    if args.resume:
        latest_ckpt = get_latest_checkpoint(args.ckpt_dir)
        if latest_ckpt:
            print(f"Found checkpoint: {latest_ckpt}")
            # safe_torch_load keeps weights_only=True in force; the numpy
            # types the RNG state needs are allowlisted, nothing else is.
            checkpoint = safe_torch_load(latest_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step']
            random.setstate(checkpoint['random_state'])
            np.random.set_state(checkpoint['np_random_state'])
            torch.set_rng_state(checkpoint['torch_rng_state'])
            print(f"RESUMING TRAINING FROM STEP {start_step}", flush=True)
        else:
            print("No checkpoint found to resume from. Starting from scratch.", flush=True)
            print(f"STARTING TRAINING FROM STEP {start_step}", flush=True)
    else:
        print(f"STARTING TRAINING FROM STEP {start_step}", flush=True)

    for step in range(start_step, args.total_steps):
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"Step {step}: Loss = {loss.item():.4f}", flush=True)
        time.sleep(0.2) # Artificially slow down to allow for disconnect simulation

        if (step + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_step_{step + 1}.pt")
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_rng_state': torch.get_rng_state(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}", flush=True)

if __name__ == "__main__":
    main()
