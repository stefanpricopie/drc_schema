#!/usr/bin/env python3
import datetime
import os
import platform
import subprocess

completed_process = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], check=True,
                                   stdout=subprocess.PIPE, universal_newlines=True)
# Strip newline character at the end
latest_git_hash = completed_process.stdout.strip()
if latest_git_hash is None:
    raise ValueError("Could not obtain the latest git hash")

# Assuming this remains constant as in your bash script
EMAIL = "stefan.pricopie@postgrad.manchester.ac.uk"
N_RUNS = 100


# Get the current date in YYYYMMDD format
current_date = datetime.datetime.now().strftime("%Y%m%d")

# Assuming BINDIR remains constant as in your bash script
BINDIR = os.path.dirname(os.path.abspath(__file__))
# Modify OUTDIR to include both the current date and the latest git hash
OUTDIR = f"logfiles/{current_date}_{latest_git_hash}"
# Ensure the directory exists
os.makedirs(OUTDIR, exist_ok=True)


def qsub_job(runner, configs, jobname, memory=None, ncores=1):
    # Generate the Config array
    config_array = "configs=(" + " \\\n         \"" + "\" \\\n         \"".join(configs) + "\")"

    # Set memory flag based on the input
    if memory is None:
        memory_flag = ""
    elif memory == 512:
        # For 32GB per core
        memory_flag = "#$ -l mem512"
    elif memory == 1500:
        # 1.5TB RAM = 48GB per core, max 32 cores (Skylake CPU). 7 nodes.
        memory_flag = "#$ -l mem1500"
    elif memory == 2000:
        # 2TB RAM   = 64GB per core, max 32 cores (Icelake CPU), 8TB SSD /tmp. 10 nodes.
        memory_flag = "#$ -l mem2000"
    else:
        raise ValueError(f"Memory value {memory} not recognised")

    # Set the number of cores
    if ncores == 1:
        ncores_flag = ""
    elif isinstance(ncores, int) and ncores > 1:
        ncores_flag = f"#$ -pe smp.pe {ncores}"
    else:
        raise ValueError(f"Number of cores {ncores} not recognised")

    # Update runner path to include experiments/ subfolder
    runner_path = f"experiments/{runner}"

    cmd = f"""#!/bin/bash --login
#$ -t 1-{len(configs)}  # Using N_RUNS to specify task range
#$ -N {jobname}
{ncores_flag}
# -l s_rt=06:00:00
{memory_flag}
# -M {EMAIL}
# -m as
#$ -cwd
#$ -j y
#$ -o {OUTDIR}

{config_array}

# Use SGE_TASK_ID to access the specific configuration
CONFIG_INDEX=$(($SGE_TASK_ID - 1))  # Arrays are 0-indexed
CONFIG=${{configs[$CONFIG_INDEX]}}

echo "{runner_path} $CONFIG"
echo "Job: $JOB_ID, Task: $SGE_TASK_ID, Config: $CONFIG"

{BINDIR}/{runner_path} $CONFIG
"""
    with subprocess.Popen(["qsub", "-v", "PATH"], stdin=subprocess.PIPE) as proc:
        proc.communicate(input=cmd.encode())


def add_config(configurations, problem, dim, algo, schema=None, delay=None):
    begin_seed = 0
    for seed in range(begin_seed, begin_seed + N_RUNS):
        if schema is None or delay is None:
            config = f"{problem} {dim} {algo} {seed}"
        else:
            config = f"{problem} {dim} {algo} {seed} --constraint {schema} --delay_factor {delay}"

        configurations.append(config)


def run_local(runner, configs):
    for i, config in enumerate(configs):
        cmd = [runner]
        cmd.extend(config.split())
        print(cmd)
        subprocess.run(cmd)


def run_job(job, job_name, memory=None, ncores=1):
    runner = "main.py"       # Your Python script for running a single experiment
    configurations = job()  # Generate the configurations for the job

    if platform.system() == "Linux":
        # assert N_RUNS == 50, "N_RUNS must be 50 for cluster runs"
        # split configurations into jobnames and configs
        qsub_job(runner=runner, configs=configurations,
                 jobname=f"{job_name}{memory if memory is not None else ''}",
                 memory=memory, ncores=ncores)
    elif platform.system() == "Darwin":  # macOS is identified as 'Darwin'
        # assert N_RUNS == 1, "N_RUNS must be 1 for local runs"
        run_local(runner=f"{os.getcwd()}/experiments/{runner}", configs=configurations)


def run():
    configurations = []

    for objective in [
        # 'ackley',
        'levy',
        # 'michalewicz',
        # 'styblinskitang',
    ]:
        for dim in [
            2,
            8,
        ]:
            bit = '0' if objective == 'michalewicz' or objective == 'levy' else '1'
            for schema in [
                # None,
                (bit*1).ljust(dim, 'N'),
                (bit*2).ljust(dim, 'N'),
            ]:
                for algo in [
                    # "ei",
                    "sobol",
                    "cmaes"
                ]:
                    for delay in [
                            4,
                            10,
                        ]:
                            add_config(configurations, problem=objective, dim=dim, algo=algo, schema=schema, delay=delay)

    return configurations


if __name__ == "__main__":
    job_mem = [
        {'job': run, 'job_name': 'levysty'},
    ]
    for job_kwargs in job_mem:
        run_job(**job_kwargs)
