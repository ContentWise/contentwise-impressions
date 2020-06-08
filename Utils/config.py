from dask.distributed import Client, LocalCluster

import os
import logging
import psutil
import sys


def configure_dask_cluster(use_processes: bool = True):
    machine_memory = psutil.virtual_memory().total
    cpu_count = psutil.cpu_count()
    partition_memory = 100 * 2 ** 20  # Recommended by Dask docs: https://docs.dask.org/en/latest/dataframe-best-practices.html#repartition-to-reduce-overhead
    dashboard_address = "localhost:8787"

    if use_processes:
        n_workers = 4  # Default value for a 16vCPU machine.
        threads_per_worker = 4  # Default value for a 16vCPU machine. Cores:= n_workers * threads_per_worker
        memory_limit = machine_memory / 4  # Each worker will have this memory limit.
    else:
        n_workers = 1  # Default value in source code
        threads_per_worker = cpu_count  # Default value in source code
        memory_limit = machine_memory

    cluster = LocalCluster(n_workers=n_workers,
                           threads_per_worker=threads_per_worker,
                           memory_limit=memory_limit,
                           processes=use_processes,
                           dashboard_address=dashboard_address)

    print(f"CPU-COUNT: {cpu_count}\nAVAILABLE-RAM: {machine_memory / 2 ** 30} GiB\nPARTITION-MEMORY: {partition_memory / 2 ** 20} MiB")

    return Client(cluster), cluster


def configure_logger(logs_dir: str, root_filename: str) -> None:
    os.makedirs(logs_dir, exist_ok=True)

    logs_filename = os.path.join(logs_dir, f"{root_filename}.log")

    log_formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(module)s|%(funcName)s|%(lineno)d|%(message)s")

    file_handler = logging.FileHandler(filename=logs_filename, mode="a")
    file_handler.setFormatter(log_formatter)

    console_out_handler = logging.StreamHandler(sys.stdout)
    console_out_handler.setFormatter(log_formatter)

    console_err_handler = logging.StreamHandler(sys.stderr)
    console_err_handler.setFormatter(log_formatter)
    console_err_handler.setLevel("ERROR")

    root_logger = logging.getLogger("contentwise-impressions")
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_out_handler)
    root_logger.addHandler(console_err_handler)
    root_logger.setLevel("DEBUG")
