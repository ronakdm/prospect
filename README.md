# prospect
Code and experiments for "Distributionally Robust Optimization with Bias and Variance Reduction".

## Abstract
We consider the distributionally robust optimization (DRO) problem with spectral risk-based uncertainty set and $f$-divergence penalty. This formulation includes common risk-sensitive learning objectives such as regularized condition value-at-risk (CVaR) and average top-$k$ loss. We present Prospect, a stochastic gradient-based algorithm that only requires tuning a single learning rate hyperparameter, and prove that it enjoys linear convergence for smooth regularized losses. This contrasts with previous algorithms that either require tuning multiple hyperparameters or potentially fail to converge due to biased gradient estimates or inadequate regularization. Empirically, we show that Prospect can converge 2-3$\times$ faster than baselines such as stochastic gradient and stochastic saddle-point methods on distribution shift and fairness benchmarks spanning tabular, vision, and language domains.

## Background
Let $\ell(w) = (\ell_1(w), \ldots, \ell_n(w))$ denote a v
We consider in this work learning problems of the form
$$
\min_{w \in \mathbb{R}^d} \max_{q \in \mathcal{P}(\sigma)} q^\top \ell(w) + \frac{\mu}{2} \|w\|_2^2 - \nu D_{\chi^2}(q\Vert \mathbf{1}_n/n),
$$
in which $\ell_i(w)$ denotes the loss of a model with weights $w \in \mathbb{R}^d$ on data instances $i$, $D_\chi^2(q\Vert \mathbf{1}_n/n) = n\|q - \mathbf{1}_n/n\|_2^2$ is the $\chi^2$-divergence between some distribution $q$ and the uniform distribution. The set $\mathcal{P}(\sigma)$ is an *uncertainty set* of possible distributions $q$ that could be shifts of the original training distribution. It is specified by some non-negative vector $\sigma \in \mathbb{R}^n$ with $\sigma_1 \leq \ldots \leq \sigma_n$ and $\sum_{i=1}^n \sigma_i = 1$. This formulation reduces to empirical risk minimization (ERM) when $\sigma_i = 1/n$ for all $i$, but contains other common risk-sensitive formulations such as the conditional value-at-risk (CVaR) or the extremile loss.

## Dependencies

The required software environment can be build and activated with Anaconda/Miniconda using the following.
```
conda env create -f environment.yml
conda activate dro
```
The environment `dro` contains the necessary packages and Python version (3.10). We recommend a hardware environment has at least 32GB CPU RAM and a GPU with at least 12GB RAM. In addition, please install PyTorch following the [installation instructions](https://pytorch.org/get-started/locally/) for your particular CUDA distribution. For example, for CUDA 11.8, run:
```
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```