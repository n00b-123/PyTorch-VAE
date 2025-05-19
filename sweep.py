import os
import yaml
import itertools
import subprocess
from copy import deepcopy

base_config_path = "configs/vae.yaml"

with open(base_config_path, "r") as f:
    base_config = yaml.safe_load(f)

# === Define hyperparameter grid ===
sweep_config = {
    "LR": [1e-3, 5e-3, 1e-2],
    "kld_weight": [0.0001, 0.00025, 0.001],
    "weight_decay": [0.0, 1e-4],
}

keys, values = zip(*sweep_config.items())
combinations = list(itertools.product(*values))

for i, combo in enumerate(combinations):
    new_config = deepcopy(base_config)
    for k, v in zip(keys, combo):
        new_config["exp_params"][k] = v

    new_config["trainer_params"]["max_epochs"] = 50
    run_name = f"lr={new_config['exp_params']['LR']}_kld={new_config['exp_params']['kld_weight']}_wd={new_config['exp_params']['weight_decay']}"
    config_path = f"configs/sweeps/sweep_{i}_{run_name}.yaml"

    # Save new config
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(new_config, f)

    print(f"\n🚀 Running config: {run_name}")
    subprocess.run(["python", "run.py", "-c", config_path])
