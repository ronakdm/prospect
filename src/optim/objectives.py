import functools
import torch
import torch.nn.functional as F
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from src.optim.smoothing import get_smooth_weights, get_smooth_weights_sorted


def squared_error_loss(w, X, y):
    return 0.5 * (y - torch.matmul(X, w)) ** 2

def squared_error_gradient(w, X, y):
    return (torch.matmul(X, w) - y)[:, None] * X

def binary_cross_entropy_loss(w, X, y):
    logits = torch.matmul(X, w)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="none"
    )
 
def binary_cross_entropy_gradient(w, X, y):
    logits = torch.matmul(X, w)
    p = 1. / (1. + torch.exp(-logits))
    return (p - y)[:, None] * X

def multinomial_cross_entropy_loss(w, X, y, n_class):
    W = w.view(-1, n_class)
    logits = torch.matmul(X, W)
    return torch.nn.functional.cross_entropy(logits, y, reduction="none")

def multinomial_cross_entropy_gradient(w, X, y, n_class):
    n = len(X)
    W = w.view(-1, n_class)
    logits = torch.matmul(X, W)
    p = torch.softmax(logits, dim=1)
    p[torch.arange(n), y] -= 1
    scores = torch.bmm(X[:, :, None], p[:, None, :])
    return scores.view(n, -1)


def get_loss(name, n_class=None):
    if name == "squared_error":
        return squared_error_loss
    elif name == "binary_cross_entropy":
        return binary_cross_entropy_loss
    elif name == "multinomial_cross_entropy":
        return lambda w, X, y: multinomial_cross_entropy_loss(w, X, y, n_class)
    else:
        raise ValueError(
            f"Unrecognized loss '{name}'! Options: ['squared_error', 'binary_cross_entropy', 'multinomial_cross_entropy']"
        )
    
def get_grad_batch(name, n_class=None):
    if name == "squared_error":
        return squared_error_gradient
    elif name == "binary_cross_entropy":
        return binary_cross_entropy_gradient
    elif name == "multinomial_cross_entropy":
        return lambda w, X, y: multinomial_cross_entropy_gradient(w, X, y, n_class)
    else:
        raise NotImplementedError


class Objective:
    def __init__(
        self,
        X,
        y,
        weight_function,
        loss="squared_error",
        l2_reg=None,
        n_class=None,
        risk_name=None,
        dataset=None,
        shift_cost=1.0,
        penalty=None,
        autodiff=True,
    ):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.weight_function = weight_function
        self.loss = get_loss(loss, n_class=n_class)
        self.grad_batch = get_grad_batch(loss, n_class=n_class)
        self.loss_name = loss
        self.n_class = n_class
        self.l2_reg = l2_reg
        self.autodiff = autodiff

        # For logging.
        self.loss_name = loss
        self.risk_name = risk_name
        self.dataset = dataset

        self.sigmas = weight_function(self.n)
        self.shift_cost = self.n * shift_cost if penalty == "l2" else shift_cost
        self.penalty = penalty

    def get_batch_loss(self, w, include_reg=True):
        with torch.no_grad():
            sorted_losses = torch.sort(self.loss(w, self.X, self.y), stable=True)[0]
            n = self.n
            if self.l2_reg:
                sm_sigmas = get_smooth_weights_sorted(
                    sorted_losses, self.sigmas, self.shift_cost, self.penalty
                )
                risk = torch.dot(
                    sm_sigmas, sorted_losses
                ) - 0.5 * self.shift_cost * torch.sum((sm_sigmas - 1 / n) ** 2)
            else:
                risk = torch.dot(self.sigmas, sorted_losses)
            if self.l2_reg and include_reg:
                risk += 0.5 * self.l2_reg * torch.norm(w) ** 2 / self.n
            return risk
        
    def get_batch_subgrad(self, w, idx=None, include_reg=True):
        if self.autodiff:
            return self.get_batch_subgrad_autodiff(w, idx=idx, include_reg=include_reg)
        else:
            return self.get_batch_subgrad_oracle(w, idx=idx, include_reg=include_reg)
    
    @torch.no_grad()
    def get_batch_subgrad_oracle(self, w, idx=None, include_reg=True):
        if idx is not None:
            X, y = self.X[idx], self.y[idx]
            sigmas = self.weight_function(len(X))
        else:
            X, y = self.X, self.y
            sigmas = self.sigmas
        sorted_losses, perm = torch.sort(self.loss(w, X, y), stable=True)
        if self.penalty:
            q = get_smooth_weights_sorted(
                sorted_losses, sigmas, self.shift_cost, self.penalty
            )
        else:
            q = sigmas
        g = torch.matmul(q, self.grad_batch(w, X, y)[perm])
        if self.l2_reg and include_reg:
            g += self.l2_reg * w.detach() / self.n
        return g

    def get_batch_subgrad_autodiff(self, w, idx=None, include_reg=True):
        if idx is not None:
            X, y = self.X[idx], self.y[idx]
            sigmas = self.weight_function(len(X))
        else:
            X, y = self.X, self.y
            sigmas = self.sigmas
        sorted_losses = torch.sort(self.loss(w, X, y), stable=True)[0]
        if self.l2_reg:
            with torch.no_grad():
                sm_sigmas = get_smooth_weights_sorted(
                    sorted_losses, sigmas, self.shift_cost, self.penalty
                )
            risk = torch.dot(sm_sigmas, sorted_losses)
        else:
            risk = torch.dot(sigmas, sorted_losses)
        g = torch.autograd.grad(outputs=risk, inputs=w)[0]
        if self.l2_reg and include_reg:
            g += self.l2_reg * w.detach() / self.n
        return g

    def get_indiv_loss(self, w, with_grad=False):
        if with_grad:
            return self.loss(w, self.X, self.y)
        else:
            with torch.no_grad():
                return self.loss(w, self.X, self.y)

    def get_indiv_prox_loss(self, w, stepsize, i):
        real_l2_reg = self.l2_reg / self.n
        with torch.no_grad():
            if self.loss_name == "squared_error":
                prox_op = prox_squared_loss
            elif self.loss_name == "multinomial_cross_entropy":
                prox_op = functools.partial(
                    prox_multinomial_log_loss_vec, n_class=self.n_class
                )
            elif self.loss_name == "binary_cross_entropy":
                prox_op = 0
            else:
                raise NotImplementedError(f"prox not implemented for {self.loss_name}")
            return prox_with_l2reg(
                w, self.X[i], self.y[i], stepsize, prox_op, real_l2_reg
            )

    def get_indiv_mor_grad(self, w, stepsize, i):
        with torch.no_grad():
            return (w - self.get_indiv_prox_loss(w, stepsize, i)) / stepsize

    @torch.no_grad()
    def get_indiv_grad(self, w, X=None, y=None):
        if not (X is None):
            return self.grad_batch(w, X, y)
        else:
            return self.grad_batch(w, self.X, self.y)

    def get_model_cfg(self):
        return {
            "objective": self.risk_name,
            "l2_reg": self.l2_reg,
            "loss": self.loss_name,
            "n_class": self.n_class,
            "shift_cost": self.shift_cost / self.n
            if self.penalty == "l2"
            else self.shift_cost,
        }


def prox_squared_loss(w, x, y, stepsize):
    scaling = stepsize / (1 + stepsize * torch.sum(x**2))
    return w - scaling * (x.squeeze().dot(w) - y) * x.squeeze()


def prox_with_l2reg(w, x, y, stepsize, prox, l2reg):
    scaled_stepsize = stepsize / (1 + stepsize * l2reg)
    scaled_w = w / (1 + stepsize * l2reg)
    return prox(scaled_w, x, y, scaled_stepsize)


def prox_multinomial_log_loss(W, x, y, stepsize, n_class):
    logits = W.mv(x.squeeze())
    sigma = torch.nn.functional.softmax(logits, dim=0)
    y_hot = torch.nn.functional.one_hot(y, n_class).squeeze()
    z_3 = torch.ones(n_class) + stepsize * torch.sum(x**2) * sigma
    z_2 = sigma / z_3
    z_1 = -y_hot / z_3 + z_2
    lam = torch.sum(z_1) / torch.sum(z_2)
    z = z_1 - lam * z_2
    return W - stepsize * torch.outer(z, x.squeeze())


def prox_binary_log_loss(w, x, y, stepsize):
    logit = w.dot(x.squeeze())
    g = -y + torch.nn.functional.sigmoid(logit)
    q = 1 / (2 * (torch.cosh(logit) + 1))
    scaling = 1 / (1 + stepsize * q * torch.sum(x**2))
    return w - stepsize * scaling * g * x.squeeze()


def prox_multinomial_log_loss_vec(w, x, y, stepsize, n_class):
    W = w.view(-1, n_class).t()
    sol = prox_multinomial_log_loss(W, x, y, stepsize, n_class)
    return sol.t().view(-1)


def get_erm_weights(n):
    return torch.ones(n, dtype=torch.float64) / n


def get_extremile_weights(n, r):
    return (
        (torch.arange(n, dtype=torch.float64) + 1) ** r
        - torch.arange(n, dtype=torch.float64) ** r
    ) / (n**r)


def get_superquantile_weights(n, q):
    weights = torch.zeros(n, dtype=torch.float64)
    idx = math.floor(n * q)
    frac = 1 - (n - idx - 1) / (n * (1 - q))
    if frac > 1e-12:
        weights[idx] = frac
        weights[(idx + 1) :] = 1 / (n * (1 - q))
    else:
        weights[idx:] = 1 / (n - idx)
    return weights


def get_esrm_weights(n, rho):
    upper = torch.exp(rho * ((torch.arange(n, dtype=torch.float64) + 1) / n))
    lower = torch.exp(rho * (torch.arange(n, dtype=torch.float64) / n))
    return math.exp(-rho) * (upper - lower) / (1 - math.exp(-rho))


def test_prox_multinomial_log_loss():
    d = 8
    k = 4
    stepsize = 1.0
    torch.manual_seed(0)
    x = torch.randn((1, d))
    y = torch.randint(0, k, (1,))
    w = torch.randn((d * k), requires_grad=True)

    v = prox_multinomial_log_loss_vec(w, x, y, stepsize, k) - w
    func = functools.partial(multinomial_cross_entropy_loss, X=x, y=y, n_class=k)
    hvp = torch.autograd.functional.hvp(func, w, v)[1]
    grad = torch.autograd.grad(func(w), w)[0]
    grad_norm_newton_prox_pb = torch.linalg.norm(stepsize * hvp + stepsize * grad + v)
    assert (
        grad_norm_newton_prox_pb < 1e-6
    ), f"First order optimality condition vioalted by {grad_norm_newton_prox_pb} "


def test_prox_binary_log_loss():
    d = 8
    stepsize = 1.0
    torch.manual_seed(0)
    x = torch.randn((1, d))
    y = torch.randint(0, 2, (1,))
    w = torch.randn((d), requires_grad=True)
    v = prox_binary_log_loss(w, x, y, stepsize) - w
    func = functools.partial(binary_cross_entropy_loss, X=x, y=y)
    hvp = torch.autograd.functional.hvp(func, w, v)[1]
    grad = torch.autograd.grad(func(w), w)[0]
    grad_norm_newton_prox_pb = torch.linalg.norm(stepsize * hvp + stepsize * grad + v)
    assert (
        grad_norm_newton_prox_pb < 1e-6
    ), f"First order optimality condition vioalted by {grad_norm_newton_prox_pb} "


if __name__ == "__main__":
    test_prox_multinomial_log_loss()
    test_prox_binary_log_loss()
