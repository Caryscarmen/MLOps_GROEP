import csv
from pathlib import Path
from typing import Any, Dict
import yaml

# TODO: Add TensorBoard Support

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: str = "experiments/results",
    ):
        # Maak een unieke map voor deze run
        self.run_dir = Path(base_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Save config to yaml in run_dir
        # Sla de volledige configuratie op
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        # Initialize CSV
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        
        # Header (TODO: add the rest of things we want to track, loss, gradients, accuracy etc.)
        self.headers = ["epoch", "train_loss", "val_loss", "val_roc_auc", "val_f2_score", "lr"]
        self.csv_writer.writerow(self.headers)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Writes metrics to CSV (and TensorBoard).
        """
        # TODO: Write other useful metrics to CSV
        # Zorg dat de volgorde overeenkomt met de headers
        row = [epoch] + [metrics.get(h, 0.0) for h in self.headers[1:]]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # TODO: Log to TensorBoard

    def get_save_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self):
        self.csv_file.close()
