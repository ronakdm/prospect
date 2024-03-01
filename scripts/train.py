"""
Train model for a particular objective and optimizer on evry hyperparameter setting.
"""

import time
import datetime
from joblib import Parallel, delayed
import sys
import argparse

L2_REG = 1.0
SHIFT_COST = 1.0
LRS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3]

# Create parser.
sys.path.append(".")
from src.utils.training import (
    OptimizationError,
    compute_training_curve,
    format_time,
    find_best_optim_cfg,
    FAIL_CODE,
)
from src.utils.io import dict_to_list
from src.utils.hyperparams import HYPERPARAM_LR

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "yacht",
        "energy",
        "simulated",
        "concrete",
        "iwildcam",
        "kin8nm",
        "power",
        "acsincome",
        "diabetes",
        "amazon",
    ],
)
parser.add_argument(
    "--objective",
    type=str,
    required=True,
    choices=[
        "extremile",
        "superquantile",
        "esrm",
        "erm",
        "extremile_lite",
        "superquantile_lite",
        "esrm_lite",
        "extremile_hard",
        "superquantile_hard",
        "esrm_hard",
    ],
)
parser.add_argument(
    "--optimizer",
    type=str,
    required=True,
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=64,
)
parser.add_argument(
    "--epoch_len",
    type=int,
    default=None,
)
parser.add_argument(
    "--use_hyperparam",
    type=int,
    default=0,
)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--n_jobs", type=int, default=-2)
args = parser.parse_args()

# Configure for input to trainers.
dataset = args.dataset
if dataset in ["yacht", "energy", "concrete", "kin8nm", "power", "acsincome"]:
    loss = "squared_error"
    n_class = None
elif dataset == "iwildcam":
    loss = "multinomial_cross_entropy"
    n_class = 60
elif dataset == "amazon":
    loss = "multinomial_cross_entropy"
    n_class = 5
elif dataset == "diabetes":
    loss = "binary_cross_entropy"

model_cfg = {
    "objective": args.objective,
    "l2_reg": L2_REG,
    "shift_cost": SHIFT_COST,
    "loss": loss,
    "n_class": n_class
}

if args.use_hyperparam:
    lrs = [HYPERPARAM_LR[args.optimizer][args.dataset][args.objective]]
else:
    lrs = LRS
optim_cfg = {
    "optimizer": args.optimizer,
    "lr": lrs,
    "epoch_len": args.epoch_len,
    "shift_cost": SHIFT_COST,
}
seeds = [1, 2]
n_epochs = args.n_epochs
parallel = bool(args.parallel)

optim_cfgs = dict_to_list(optim_cfg)

config = {
    "dataset": dataset,
    "model_cfg": model_cfg,
    "optim_cfg": optim_cfg,
    "parallel": parallel,
    "seeds": seeds,
    "n_epochs": n_epochs,
    "epoch_len": args.epoch_len,
}

# Display.
print("-----------------------------------------------------------------")
for key in config:
    print(f"{key}:" + " " * (16 - len(key)), config[key])
print(f"Start:" + " " * 11, {str(datetime.datetime.now())})
print("-----------------------------------------------------------------")


# Run optimization.
def worker(optim):
    name, lr = optim["optimizer"], optim["lr"]
    diverged = False
    for seed in seeds:
        code = compute_training_curve(
            dataset,
            model_cfg,
            optim,
            seed,
            n_epochs,
        )
        if code == FAIL_CODE:
            diverged = True
    if diverged:
        print(f"Optimizer '{name}' diverged at learning rate {lr}!")


tic = time.time()
if parallel:
    Parallel(n_jobs=args.n_jobs)(delayed(worker)(optim) for optim in optim_cfgs)
else:
    for optim in optim_cfgs:
        worker(optim)
toc = time.time()
print(f"Time:         {format_time(toc-tic)}.")

# Save best configuration.
find_best_optim_cfg(dataset, model_cfg, optim_cfgs, seeds)
