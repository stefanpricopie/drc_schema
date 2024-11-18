import itertools
import os
import re
from collections import defaultdict

import pandas as pd
import torch


def find_pt_files(folder_path, filter=None):
    pt_files = []
    # List only files in the specified folder, not in subdirectories
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.endswith(".pt"):
            if filter is None:
                pt_files.append(file_path)
            elif filter=='constrained':
                if 'delay' in file:
                    pt_files.append(file_path)
            elif re.fullmatch(r'^[01N]+$', filter):
                if f"_{filter}_" in file:
                    pt_files.append(file_path)
            else:
                raise ValueError(f"Invalid filter: {filter}. Must be 'constrained' or a sequence of '0', '1', or 'N' or None")
    return pt_files


def group_files_by_config(files):
    # Regular expression to match the parts of the filename
    # Explanation:
    # - `(\d+)D`: Captures the problem number (e.g., `2`)
    # - `_([^\_]+)_`: Captures the identifier (e.g., `ei`) until the next underscore
    # - `(?:([01N]+)_)?`: Optionally captures a sequence of "0", "1", or "N" followed by an underscore
    # - `(?:delay(\d+)_)?`: Optionally captures the "delay" part followed by a number
    # - `\d{4}\.pt`: Matches exactly 4 digits (the seed), followed by `.pt`
    regex = r"(\d+)D_([^\_]+)_(?:([01N]+)_)?(?:delay(\d+)_)?\d{4}\.pt"

    # Regular expression to match the parts of the filename
    pattern = re.compile(regex)

    # Dictionary to store files by their group
    files_by_group = defaultdict(list)

    # Extract groups from each filename and associate it with the file
    for filepath in files:
        filename = re.split(r'/', filepath)[-1]  # Extract the filename from the path
        match = pattern.match(filename)

        if match:
            # Create a tuple of the captured groups
            dim = match.group(1)
            identifier = match.group(2)
            num = match.group(3)  # Could be None if not present
            delay = match.group(4)  # Could be None if not present

            # Group key as tuple of extracted values
            group = (dim, identifier, num, delay)

            # Add the filename to the appropriate group in the dictionary
            files_by_group[group].append(filepath)
        else:
            raise ValueError(f"Filename {filename} does not match the expected pattern")

    # Now, print the files for each unique group
    for group, files in files_by_group.items():
        print(f"Group: {group}: {len(files)} files")

    return files_by_group

def dataframe_from_group(problem, key, key_files):
    dim, algo, schema, delay = key

    df = pd.DataFrame()
    for f in key_files:
        try:
            # Load the data
            data = torch.load(f, weights_only=False)
        except Exception as e:
            print(f"Error loading file: {f}")
            raise e

        # Extract the seed from the filename
        seed = int(re.search(r'\d{4}', f).group())

        # Number of evaluations (excluding initial evaluations)
        n_evaluations_plus_one = len(data['time_constraints']) + 1

        # Create a dataframe for this specific run
        run_df = pd.DataFrame({
            **{f'X{i+1}': x_dim[-n_evaluations_plus_one:] for i, x_dim in enumerate(data['X'].T)},  # Extract each dimension as a column
            'Y': -data['Y'][-n_evaluations_plus_one:].squeeze().numpy(),  # Exclude first n_init-1 evaluations
            'min_Y': -data['best_objs'].squeeze().numpy(),   # Includes the best of the initial evaluations. Length is n_evaluations+1
            'time_constraints': [None] + data['time_constraints'],  # Include first a None for the last of the initial evaluations
            'algo': algo,
            'algo_kwargs': algo if schema is None else f"{algo}_constrained-{schema[:2]}-sticky{delay}",
            'seed': seed,
            'schema': schema,
            'delay': delay,
            'problem': f"{problem.capitalize()}",
            'dim': dim,
            'problem_dim': f"{problem.capitalize()}{dim}D",
        })

        if schema is not None:
            constraint_lb = torch.tensor([0.5 if c == '1' else 0. for c in schema],)
            constraint_ub = torch.tensor([0.5 if c == '0' else 1. for c in schema],)
            constraint_bounds = torch.stack([constraint_lb, constraint_ub], dim=0)

            # Check if each dimension satisfies the bounds for each row
            satisfies_lower_bound = data['X'] >= constraint_bounds[0]
            satisfies_upper_bound = data['X'] <= constraint_bounds[1]

            # Combine the conditions across all dimensions (check if all are True for each row)
            in_schema = torch.all(satisfies_lower_bound & satisfies_upper_bound, dim=1)
    
            run_df['in_schema'] = in_schema[-n_evaluations_plus_one:].numpy()  # Exclude first n_init-1 evaluations

        # Name the index
        run_df.index.name = 'T'

        # Append this run's data to the cumulative dataframe
        df = pd.concat([df, run_df.reset_index()])

    # Ensure the processed_data directory exists
    processed_data_path = os.path.abspath('./processed_data')
    os.makedirs(processed_data_path, exist_ok=True)

    # Construct the file name using experiment, algorithm, and configuration
    output_filename = f'{problem}_{dim}_{algo}_{schema}_{delay}.csv'

    # Create the full path to save the CSV
    full_path = os.path.join(processed_data_path, output_filename)

    # Save the DataFrame as a CSV file
    df.to_csv(full_path, index=False)
    print(f"Data saved to {full_path}")


def delay_tensors(df_tensor, schema, delay):
    constraint_lb = torch.tensor([0.5 if c == '1' else 0. for c in schema],)
    constraint_ub = torch.tensor([0.5 if c == '0' else 1. for c in schema],)
    constraint_bounds = torch.stack([constraint_lb, constraint_ub], dim=0)

    # Check if each dimension satisfies the bounds for each row
    satisfies_lower_bound = df_tensor >= constraint_bounds[0]
    satisfies_upper_bound = df_tensor <= constraint_bounds[1]

    # Combine the conditions across all dimensions (check if all are True for each row)
    in_schema = torch.all(satisfies_lower_bound & satisfies_upper_bound, dim=1)

    time_constrained = delay if in_schema[0] else 0
    dts = [0]    # 0-indexed is the last initial evaluation
    time_constraints = [None]


    for i in range(1, len(in_schema)):
        if time_constrained > 0 and in_schema[i]:
            # Decrease number of evaluations left to perform under the constrained region
            dts.append(1)    # costs one time step
            time_constrained -= 1
        elif time_constrained == 0 and in_schema[i]:
            dts.append(1)
            time_constrained = delay
        elif time_constrained > 0 and not in_schema[i]:
            dts.append(time_constrained+1)
            time_constrained = 0
        else:
            dts.append(1)
        time_constraints.append(time_constrained)

    return torch.tensor(dts), time_constraints, in_schema


if __name__ == '__main__':
    # Define the possible values for each parameter
    problems = [
        # 'ackley',
        'levy',
        # 'michalewicz',
        # 'styblinskitang',
    ]
    dimensions = [
        2,
        8
    ]
    algorithms = [
        'cmaes',
        'sobol',
        'ei'
    ]

    # Get all combinations of problems, dimensions, and algorithms
    combinations = list(itertools.product(problems, dimensions, algorithms))

    # Print the combinations
    for problem, dim, algo in combinations:
        experiments_path = f'./experiments/{problem}/{dim}D/{algo}'

        # Use the function from utils to find the .pt files
        files = find_pt_files(experiments_path)

        # Group the files by their configuration
        run_configs = group_files_by_config(files)

        # Create a DataFrame for each group of files
        for key, key_files in run_configs.items():
            dataframe_from_group(problem, key, key_files)