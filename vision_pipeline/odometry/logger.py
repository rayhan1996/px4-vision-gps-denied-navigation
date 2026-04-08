import csv
from pathlib import Path
from datetime import datetime

class ExperimentLogger:
    def __init__(self, base_dir="../experiments", experiment_name="vo_run"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.exp_dir / "trajectory.csv"

        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.writer = csv.writer(self.csv_file)

        # Header (PX4-friendly format)
        self.writer.writerow([
            "timestamp",
            "tx", "ty", "tz",
            "qx", "qy", "qz", "qw"
        ])

        print(f"📁 Experiment folder: {self.exp_dir}")

    def log(self, timestamp, t, q):
        self.writer.writerow([
            timestamp,
            t[0], t[1], t[2],
            q[0], q[1], q[2], q[3]
        ])

    def close(self):
        self.csv_file.close()
