#!/usr/bin/env python3
"""Utilities for running experiments."""
from typing import Optional, Tuple, Union

import torch
from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from botorch.test_functions import SyntheticTestFunction
from botorch.test_functions.synthetic import (
    Ackley, Beale, Branin, Bukin, DixonPrice, DropWave, EggHolder, Griewank,
    Hartmann, HolderTable, Levy, Michalewicz, Rastrigin, Rosenbrock, Shekel, StyblinskiTang
)
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor
from dynamic_resource_constraints.problems.synthetic import GoldsteinPrice, Salomon, Schwefel, Shubert



def eval_problem(X: Tensor, base_function: SyntheticTestFunction) -> Tensor:
    # assert X is in unit cube
    assert (X >= 0).all() and (X <= 1).all()
    # normalize from integers to unit cube
    X_numeric = unnormalize(X, base_function.bounds)
    Y = base_function(X_numeric)
    if Y.ndim == X_numeric.ndim - 1:
        Y = Y.unsqueeze(-1)
    return Y


def generate_initial_data(
    n: int,
    base_function: SyntheticTestFunction,
    bounds: Tensor,
    tkwargs: dict,
) -> Tuple[Tensor, Tensor]:
    r"""
    Generates the initial data for the experiments.
    Args:
        n: Number of training points.
        base_function: The base problem.
        bounds: The bounds to generate the training points from. `2 x d`-dim tensor.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        The train_X and train_Y. `n x d` and `n x m`.
    """
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(-2).to(**tkwargs)
    train_obj = eval_problem(train_x, base_function=base_function)
    return train_x, train_obj


def initialize_model(
    train_x: Tensor,
    train_y: Tensor,
) -> Tuple[
    Union[ExactMarginalLogLikelihood, SumMarginalLogLikelihood],
    Union[FixedNoiseGP, SingleTaskGP, ModelListGP],
]:
    r"""Constructs the model and its MLL.

    Args:
        train_x: An `n x d`-dim tensor of training inputs.
        train_y: An `n x m`-dim tensor of training outcomes.
    Returns:
        The model and the MLL.
    """
    # define the model for objective
    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=torch.full_like(train_y, 1e-6),
    ).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def get_problem(name: str, dim: int, **kwargs) -> SyntheticTestFunction:
    r"""Initialize the test function.

    Args:
        name: The name of the test function.
        dim: The dimension of the test function.
        kwargs: Additional arguments for the test function.
    """
    bounds = kwargs.get("bounds", None)  # Get bounds or None if not specified
    if bounds is not None and torch.tensor(bounds).ndim == 1:
        lb, ub = bounds
        bounds = [(lb, ub) for _ in range(dim)]

    OBJECTIVES = {
        'ackley': Ackley,
        # 'beale': Beale,
        # 'branin': Branin,
        # 'bukin': Bukin,
        # 'dixonprice': DixonPrice,
        # 'dropwave': DropWave,
        # 'eggholder': EggHolder,
        # 'goldsteinprice': GoldsteinPrice,
        # 'griewank': Griewank,
        # 'holdertable': HolderTable,
        'levy': Levy,
        'michalewicz': Michalewicz,
        # 'rastrigin': Rastrigin,
        # 'rosenbrock': Rosenbrock,
        # 'salomon': Salomon,
        # 'schwefel': Schwefel,
        # 'shekel': Shekel,
        # 'shubert': Shubert,
        'styblinskitang': StyblinskiTang,
        # Add other objectives here
    }

    obj = OBJECTIVES[name]
    if 'dim' in obj.__init__.__code__.co_varnames:
        return OBJECTIVES[name](dim=dim, negate=True, bounds=bounds,)
    elif isinstance(obj, Shekel):
        # Shekel has a different parameter name
        return obj(m=dim, negate=True, bounds=bounds,)
    else:
        return OBJECTIVES[name](negate=True, bounds=bounds,)