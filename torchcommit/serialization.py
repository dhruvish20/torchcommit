import os
import json
import torch

BASE_DIR = ".torchcommit"


def generate_run_id(prefix="run"):
    os.makedirs(BASE_DIR, exist_ok=True)
    existing = [
        name for name in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, name)) and name.startswith(prefix)
    ]

    numbers = []
    for name in existing:
        try:
            num = int(name.replace(f"{prefix}_", ""))
            numbers.append(num)
        except ValueError:
            continue

    next_id = max(numbers, default=0) + 1
    return f"{prefix}_{next_id}"


def create_run_folder(run_id: str) -> str:
    run_folder = os.path.join(BASE_DIR, run_id)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def save_json(data: dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_model(model, path: str, enabled: bool = True):
    if enabled:
        torch.save(model.state_dict(), path)


def save_optimizer(optimizer, path: str, enabled: bool = True):
    if enabled:
        torch.save(optimizer.state_dict(), path)


def save_metrics(logs: list, path: str):
    with open(path, 'w') as f:
        json.dump(logs, f, indent=4)
