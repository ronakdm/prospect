import pandas as pd
import time
from tqdm import tqdm

from src.optim.baselines import (
    StochasticSubgradientMethod,
    StochasticRegularizedDualAveraging,
    SmoothedLSVRG,
    SaddleSAGA,
)
from src.optim.prospect import Prospect, ProspectMoreau
from src.optim.objectives import (
    Objective,
    get_extremile_weights,
    get_superquantile_weights,
    get_esrm_weights,
    get_erm_weights,
)


class OptimizationError(RuntimeError):
    pass


def train_model(optimizer, val_objective, n_epochs):
    epoch_len = optimizer.get_epoch_len()
    metrics = [compute_metrics(-1, optimizer, val_objective, 0.0)]
    penalty = optimizer.objective.penalty
    init_loss = metrics[0]["train_loss"] + penalty  # makes this positive

    for epoch in tqdm(range(n_epochs)):
        tic = time.time()
        optimizer.start_epoch()
        for _ in range(epoch_len):
            optimizer.step()
        optimizer.end_epoch()
        toc = time.time()

        # Logging.
        metrics.append(compute_metrics(epoch, optimizer, val_objective, toc - tic))
        if metrics[-1]["train_loss"] + penalty >= 1.5 * init_loss:
            raise OptimizationError(
                f"train loss 50% greater than inital loss! (epoch {epoch})"
            )

    result = {
        "weights": optimizer.weights,
        "metrics": pd.DataFrame(metrics),
    }
    return result


def get_optimizer(optim_cfg, objective, seed, device="cpu"):
    name, lr, epoch_len, shift_cost = (
        optim_cfg["optimizer"],
        optim_cfg["lr"],
        optim_cfg["epoch_len"],
        optim_cfg["shift_cost"],
    )

    lrd = 0.5 if "lrd" not in optim_cfg.keys() else optim_cfg["lrd"]
    penalty = "l2"
    if name == "sgd":
        return StochasticSubgradientMethod(
            objective, lr=lr, seed=seed, epoch_len=epoch_len
        )
    elif name == "srda":
        return StochasticRegularizedDualAveraging(
            objective, lr=lr, seed=seed, epoch_len=epoch_len
        )
    elif name == "lsvrg":
        return SmoothedLSVRG(
            objective,
            lr=lr,
            smooth_coef=shift_cost,
            smoothing=penalty,
            seed=seed,
            length_epoch=epoch_len,
        )
    elif name == "saddlesaga":
        # best lr for V1.
        return SaddleSAGA(
            objective,
            lrp=lr,
            lrd=lr / 10,
            smoothing=penalty,
            sm_coef=shift_cost,
            seed_grad=seed,
            seed_table=3 * seed,
            epoch_len=epoch_len,
        )
    elif name == "prospect":
        return Prospect(
            objective,
            lrp=lr,
            epoch_len=epoch_len,
            shift_cost=shift_cost,
            penalty=penalty,
            seed_grad=seed,
            seed_table=3 * seed,
        )
    elif name == "moreau":
        return ProspectMoreau(
            objective,
            lr=lr,
            penalty=penalty,
            shift_cost=shift_cost,
            seed_grad=seed,
            seed_table=3 * seed,
            epoch_len=epoch_len,
            device=device,
        )
    else:
        raise ValueError("Unreocgnized optimizer!")


def get_objective(model_cfg, X, y, dataset=None):
    name, l2_reg, loss, n_class, shift_cost = (
        model_cfg["objective"],
        model_cfg["l2_reg"],
        model_cfg["loss"],
        model_cfg["n_class"],
        model_cfg["shift_cost"],
    )
    if name == "erm":
        weight_function = lambda n: get_erm_weights(n)
    elif name == "extremile":
        weight_function = lambda n: get_extremile_weights(n, 2.0)
    elif name == "superquantile":
        weight_function = lambda n: get_superquantile_weights(n, 0.5)
    elif name == "esrm":
        weight_function = lambda n: get_esrm_weights(n, 1.0)
    elif name == "extremile_lite":
        weight_function = lambda n: get_extremile_weights(n, 1.5)
    elif name == "superquantile_lite":
        weight_function = lambda n: get_superquantile_weights(n, 0.25)
    elif name == "esrm_lite":
        weight_function = lambda n: get_esrm_weights(n, 0.5)
    elif name == "extremile_hard":
        weight_function = lambda n: get_extremile_weights(n, 2.5)
    elif name == "superquantile_hard":
        weight_function = lambda n: get_superquantile_weights(n, 0.75)
    elif name == "esrm_hard":
        weight_function = lambda n: get_esrm_weights(n, 2.0)

    return Objective(
        X,
        y,
        weight_function,
        l2_reg=l2_reg,
        loss=loss,
        n_class=n_class,
        risk_name=name,
        dataset=dataset,
        shift_cost=shift_cost,
        penalty="l2",
    )


def compute_metrics(epoch, optimizer, val_objective, elapsed):
    return {
        "epoch": epoch,
        "train_loss": optimizer.objective.get_batch_loss(optimizer.weights).item(),
        "train_loss_unreg": optimizer.objective.get_batch_loss(
            optimizer.weights, include_reg=False
        ).item(),
        "val_loss": val_objective.get_batch_loss(optimizer.weights).item(),
        "elapsed": elapsed,
    }
