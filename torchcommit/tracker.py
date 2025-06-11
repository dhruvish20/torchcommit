import time
import torch
from torchcommit.serialization import (
    generate_run_id,
    create_run_folder,
    save_json,
    save_model,
    save_optimizer,
    save_metrics
)


class ExperimentTracker:
    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        config=None,
        scheduler=None,
        checkpoints=True,
        dataloader=None,
        epochs=None,
        early_stopping_callback=None  # New: for auto-detecting patience
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.checkpoints = checkpoints

        self.start_time = None
        self.end_time = None
        self.logs = []

        # Run folder setup
        self.run_id = generate_run_id()
        self.run_path = create_run_folder(self.run_id)

        # Smart config setup
        self.config = config or {}

        # Try to extract global PyTorch seed
        try:
            self.config["torch_seed"] = torch.initial_seed()
        except Exception:
            self.config["torch_seed"] = "unknown"

        # Optionally add early stopping info
        if early_stopping_callback:
            self.config["early_stopping"] = True
            self.config["early_stopping_patience"] = getattr(early_stopping_callback, "patience", "unknown")
        else:
            self.config["early_stopping"] = False

        # Optionally track epochs
        if epochs is not None:
            self.config["epochs"] = epochs

        # Optionally track DataLoader config
        if dataloader is not None:
            self.config["dataloader"] = {
                "batch_size": dataloader.batch_size,
                "shuffle": getattr(dataloader, "shuffle", "unknown"),
                "num_workers": dataloader.num_workers,
                "drop_last": dataloader.drop_last,
                "pin_memory": dataloader.pin_memory,
                "dataset": type(dataloader.dataset).__name__,
                "dataset_size": len(dataloader.dataset)
            }

    def start(self):
        self.start_time = time.time()
        self.logs.append({"event": "start", "timestamp": self.start_time})

        base_config = {
            "architecture": self._extract_model_architecture(),
            "loss_function": self._extract_loss_function(),
            "optimizer": self._extract_optimizer_config(),
            "user_config": self.config
        }

        save_json(base_config, f"{self.run_path}/config.json")

    def log(self, epoch, metrics: dict):
        timestamp = time.time()
        log_entry = {
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": metrics
        }
        self.logs.append(log_entry)

    def end(self):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        self.logs.append({"event": "end", "timestamp": self.end_time})
        self.logs.append({"event": "duration", "seconds": duration})

        # Save training metrics
        save_metrics(self.logs, f"{self.run_path}/metrics.json")

        # Save summary metadata
        meta = {
            "run_id": self.run_id,
            "duration_sec": duration,
            "total_epochs": self._extract_epoch_count()
        }
        save_json(meta, f"{self.run_path}/meta.json")

        if self.checkpoints:
            save_model(self.model, f"{self.run_path}/model.pth")
            save_optimizer(self.optimizer, f"{self.run_path}/optimizer.pth")

    # --- Internal Extraction Methods ---

    def _extract_loss_function(self):
        return type(self.loss_function).__name__

    def _extract_optimizer_config(self):
        config = {
            "type": type(self.optimizer).__name__,
        }

        if self.optimizer.param_groups:
            for key, value in self.optimizer.param_groups[0].items():
                if isinstance(value, (float, int, str, bool)):
                    config[key] = value

        return config

    def _extract_model_architecture(self):
        architecture = {
            "model_class": type(self.model).__name__,
            "layer_count": 0,
            "layers": [],
            "total_parameters": sum(p.numel() for p in self.model.parameters())
        }

        for name, module in self.model.named_modules():
            if name == "":
                continue
            layer_info = {
                "name": name,
                "type": type(module).__name__,
                "params": sum(p.numel() for p in module.parameters() if p.requires_grad)
            }
            architecture["layers"].append(layer_info)

        architecture["layer_count"] = len(architecture["layers"])
        return architecture

    def _extract_epoch_count(self):
        epoch_nums = [entry["epoch"] for entry in self.logs if "epoch" in entry]
        return max(epoch_nums) + 1 if epoch_nums else 0
