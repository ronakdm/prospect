# Prospect
Code and experiments for the paper ["Distributionally Robust Optimization with Bias and Variance Reduction"](https://openreview.net/pdf?id=TTrzgEZt9s) published in **ICLR 2024** (Spotlight).

## Abstract
We consider the distributionally robust optimization (DRO) problem with spectral risk-based uncertainty set and $f$-divergence penalty. This formulation includes common risk-sensitive learning objectives such as regularized condition value-at-risk (CVaR) and average top $k$ loss. We present Prospect, a stochastic gradient-based algorithm that only requires tuning a single learning rate hyperparameter, and prove that it enjoys linear convergence for smooth regularized losses. This contrasts with previous algorithms that either require tuning multiple hyperparameters or potentially fail to converge due to biased gradient estimates or inadequate regularization. Empirically, we show that Prospect can converge 2-3 times faster than baselines such as stochastic gradient and stochastic saddle-point methods on distribution shift and fairness benchmarks spanning tabular, vision, and language domains.

## Background
We consider in this work learning problems of the form

$$
\min_{w \in \mathbb{R}^d} \max_{q \in \mathcal{P}(\sigma)} q^\top \ell(w) + \frac{\mu}{2} \Vert w\Vert_2^2 - \nu D_{\chi^2}(q\Vert \mathbf{1}_n/n),
$$

in which $\ell_i(w)$ denotes the loss of a model with weights $w \in \mathbb{R}^d$ on data instances $i$, $D_{\chi^2}(q\Vert \mathbf{1}_n/n) = n \Vert q - \mathbf{1}_n/n \Vert_2^2$ is the ${\chi}^2$-divergence between some distribution $q$ and the uniform distribution. The set $\mathcal{P}(\sigma)$ is an *uncertainty set* of possible distributions $q$ that could be shifts of the original training distribution. It is specified by some non-negative vector $\sigma \in \mathbb{R}^n$ with $0 \leq \sigma_1 \leq \ldots \leq \sigma_n$ and $\sigma_1 + \ldots + \sigma_n = 1$. This formulation reduces to empirical risk minimization (ERM) when $\sigma_i = 1/n$ for all $i$, but contains other common risk-sensitive formulations such as the conditional value-at-risk (CVaR) or the extremile loss.

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

## Datasets

Of the nine datasets used in the paper, five (`yacht`, `energy`, `concrete`, `kin8nm`, `power`) are downloaded automatically when the are loaded (see `tutorial.ipynb`). The `iwildcam` dataset requires a larger amount of processing to reproduce fully, and the preprocessed version of this dataset is already included in the `data` folder. The fairness benchmark datasets `acsincome` and `diabetes` can be prepared by running:
```
python scripts/download_acsincome.py
python scripts/download_diabetes.py
```
Finally, reproducing the `amazon` dataset requires fine-tuning a BERT model to produce frozen feature representations. This can be done by running through the entirety of the `download_amazon.ipynb` notebook. A preprocessed version already exists in the `data/amazon` folder.

## Tutorial

After completing all of the above steps, see `tutorial.ipynb` for a walkthrough of the code structure and how to reproduce experimental results.

## Reproducing Figures from the Paper

The figures from the main text, namely the experiments on tabular regression, fairness benchmarks, and text/image distribution shift benchmarks can be reproduced by running through the notebooks `regression.ipynb`, `fairness.ipynb`, `amazon.ipynb`, and `iwildcam.ipynb`, respectively.

## References
If you found this repository useful, please consider citing the following papers:

```
@inproceedings{mehta2024distributionally,
  title={{Distributionally Robust Optimization with Bias and Variance Reduction}},
  author={Mehta, Ronak and Roulet, Vincent and Pillutla, Krishna and Harchaoui, Zaid},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{mehta2023stochastic,
  title={{Stochastic Optimization for Spectral Risk Measures}},
  author={Mehta, Ronak and Roulet, Vincent and Pillutla, Krishna and Liu, Lang and Harchaoui, Zaid},
  booktitle={AISTATS},
  pages={10112--10159},
  year={2023},
}

```
