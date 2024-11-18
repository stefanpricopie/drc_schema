#!/usr/bin/env python3
r"""
The main script for running a single replication.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict

import torch

from dynamic_resource_constraints.run_one_replication import run_one_replication


# Function to parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Handle constrained and unconstrained problems.")

    problem_labels = [
        "ackley_og",    # Ackley default bounds
        "ackley",       # Ackley with  bounds = [(-32.768 / 4, 32.768 / 2) for _ in range(dim)] (smaller, asymmetric bounds)
        "michalewicz",
        "levy",
        "styblinskitang",
    ]

    # Required arguments
    parser.add_argument("problem", type=str, choices=problem_labels, help="The problem name or identifier.")
    parser.add_argument("dim", type=int, help="Dimension of the problem.")
    parser.add_argument("label", type=str, help="Label for the experiment.")
    parser.add_argument("seed", type=int, help="Random seed for reproducibility.")

    # Optional arguments for constrained case
    parser.add_argument("--constraint", type=str, help="Schema constraint vector (e.g., '01N') if constrained.")
    parser.add_argument("--delay_factor", type=int, help="Delay factor for constrained problem.")

    return parser.parse_args()

def fetch_data(kwargs: Dict[str, Any]) -> None:
    # this modifies kwargs in place
    problem_kwargs = kwargs.get("problem_kwargs", {})
    key = problem_kwargs.get("datapath")

    if key is not None:
        data = torch.load(key)
        problem_kwargs["data"] = data
        kwargs["problem_kwargs"] = problem_kwargs


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Parse the command-line arguments
    args = parse_args()

    problem: str = args.problem
    exp_dir = os.path.join(current_dir, problem)

    dim: int = args.dim
    label: str = args.label
    seed: int = args.seed

    # Initialize constraint and delay for file naming
    extra_title = ""
    constraint: str = args.constraint
    delay_factor: int = args.delay_factor

    # Handle constrained and unconstrained cases
    if constraint and delay_factor is not None:
        # Constrained case: Validate constraint and delay factor

        # Assert length of constraint matches the dimension
        assert len(constraint) == dim, f"Constraint length ({len(constraint)}) must match the dimension ({dim})"

        # Assert that constraint contains only '0', '1', or 'N'
        assert all(c in ["0", "1", "N"] for c in constraint), "Constraint must contain only '0', '1', or 'N'."

        # Add constraint and delay to the title for constrained problems
        extra_title = f"_{constraint}_delay{delay_factor}"

    # Construct the output path, appending constraint and delay factor if present
    output_path = os.path.join(
        exp_dir,
        f"{dim}D",
        label,
        f"{dim}D_{label}{extra_title}_{str(seed).zfill(4)}.pt"
    )

    print(f"Output path: {output_path}")

    # Create the output directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Skip experiment if output file already exists
    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}. Skipping experiment.")
        sys.exit(0)

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    save_callback = lambda data: torch.save(data, output_path)
    save_frequency = 5
    fetch_data(kwargs=kwargs)
    run_one_replication(
        seed=seed,
        dim=dim,
        label=label,
        constraint=constraint,
        delay_factor=delay_factor,
        save_callback=save_callback,
        save_frequency=save_frequency,
        **kwargs,   # function_name, batch_size
    )
