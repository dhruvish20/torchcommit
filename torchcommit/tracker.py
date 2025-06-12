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
        early_stopping_callback=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.checkpoints = checkpoints

        self.start_time = None
        self.end_time = None
        self.logs = []

        self.run_id = generate_run_id()
        self.run_path = create_run_folder(self.run_id)

        self.config = config or {}

        try:
            self.config["torch_seed"] = torch.initial_seed()
        except Exception:
            self.config["torch_seed"] = "unknown"

        if early_stopping_callback:
            self.config["early_stopping"] = True
            self.config["early_stopping_patience"] = getattr(early_stopping_callback, "patience", "unknown")
        else:
            self.config["early_stopping"] = False

        if epochs is not None:
            self.config["epochs"] = epochs

        if dataloader is not None:
            self.config["dataloader"] = self._extract_dataloader_info(dataloader)

    def start(self):
        if self.start_time is not None:
            raise RuntimeError("Tracker has already been started.")

        self.start_time = time.time()
        self.logs.append({"event": "start", "timestamp": self.start_time})

        config_data = {
            "architecture": self._extract_model_architecture(),
            "loss_function": self._extract_loss_function(),
            "optimizer": self._extract_optimizer_config(),
            "scheduler": self._extract_scheduler_config(),
            "user_config": self.config
        }

        save_json(config_data, f"{self.run_path}/config.json")

    def log(self, epoch, metrics: dict):
        if self.start_time is None:
            raise RuntimeError("Call tracker.start() before logging.")

        timestamp = time.time()
        self.logs.append({
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": metrics
        })

    def end(self):
        if self.start_time is None:
            raise RuntimeError("Call tracker.start() before tracker.end().")

        if self.end_time is not None:
            return  # Already ended

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        self.logs.append({"event": "end", "timestamp": self.end_time})
        self.logs.append({"event": "duration", "seconds": duration})

        # Save logs and summary
        save_metrics(self.logs, f"{self.run_path}/metrics.json")

        meta = {
            "run_id": self.run_id,
            "duration_sec": duration,
            "total_epochs": self._extract_epoch_count()
        }
        save_json(meta, f"{self.run_path}/meta.json")

        if self.checkpoints:
            save_model(self.model, f"{self.run_path}/model.pth")
            save_optimizer(self.optimizer, f"{self.run_path}/optimizer.pth")


    def _extract_loss_function(self):
        return type(self.loss_function).__name__

    def _extract_optimizer_config(self):
        config = {"type": type(self.optimizer).__name__}
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
    
    def _extract_scheduler_config(self):
        if self.scheduler is None:
            return None

        config = {
            "type": type(self.scheduler).__name__,
        }

        try:
            for key, value in self.scheduler.__dict__.items():
                if isinstance(value, (float, int, str, bool)):
                    config[key] = value
        except Exception:
            pass

        return config


    def _extract_dataloader_info(self, dataloader):
        return {
            "batch_size": dataloader.batch_size,
            "shuffle": getattr(dataloader, "shuffle", "unknown"),
            "num_workers": dataloader.num_workers,
            "drop_last": dataloader.drop_last,
            "pin_memory": dataloader.pin_memory,
            "dataset": type(dataloader.dataset).__name__,
            "dataset_size": len(dataloader.dataset)
        }

    def _extract_epoch_count(self):
        epoch_nums = [entry["epoch"] for entry in self.logs if "epoch" in entry]
        return max(epoch_nums) + 1 if epoch_nums else 0
