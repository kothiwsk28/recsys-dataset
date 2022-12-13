import subprocess
import random

# Automated random search experiments
num_exps = 4
random.seed(0)

for _ in range(num_exps):
    params = {
        "data_fraction": random.choice([0.0001]),
        "hours_cutoff": random.choice([18, 12, 24, 36, 48]),
        "weights": random.choice([[1, 1, 1]])

    }
    subprocess.run(["dvc", "exp", "run", "--queue",
                    "--set-param", f"co-visitation.data_fraction={params['data_fraction']}",
                    "--set-param", f"co-visitation.hours_cutoff={params['hours_cutoff']}",
                    "--set-param", f"co-visitation.weights={params['weights']}"])