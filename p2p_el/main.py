import logging
from pathlib import Path
from shutil import copy
import argparse
import datetime

from localconfig import LocalConfig
from torch import multiprocessing as mp

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Linear import Linear
from EL_Local import EL_Local


def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(dict(config.items("DATASET")))
    return config

def get_args():
    """
    Utility to parse arguments.

    Returns
    -------
    args
        Command line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-mid", "--machine_id", type=int, default=0)
    parser.add_argument("-ps", "--procs_per_machine", type=int, default=1, nargs="+")
    parser.add_argument("-ms", "--machines", type=int, default=1)
    parser.add_argument(
        "-ld",
        "--log_dir",
        type=str,
        default="./{}".format(datetime.datetime.now().isoformat(timespec="minutes")),
    )
    parser.add_argument(
        "-wsd",
        "--weights_store_dir",
        type=str,
        default="./{}_ws".format(datetime.datetime.now().isoformat(timespec="minutes")),
    )
    parser.add_argument("-is", "--iterations", type=int, default=1)
    parser.add_argument("-cf", "--config_file", type=str, default="config.ini")
    parser.add_argument("-ll", "--log_level", type=str, default="INFO")
    parser.add_argument("-gf", "--graph_file", type=str, default="36_nodes.edges")
    parser.add_argument("-gt", "--graph_type", type=str, default="edges")
    parser.add_argument("-ta", "--test_after", type=int, default=5)
    parser.add_argument("-tea", "--train_evaluate_after", type=int, default=1)
    parser.add_argument("-ro", "--reset_optimizer", type=int, default=1)
    parser.add_argument("-sm", "--server_machine", type=int, default=0)
    parser.add_argument("-sr", "--server_rank", type=int, default=-1)
    parser.add_argument("-wr", "--working_rate", type=float, default=1.0)
    parser.add_argument("--local-only", action="store_true", help="Disable gossip exchange")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    log_level = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    config = read_ini(args.config_file)
    my_config = dict()
    for section in config:
        my_config[section] = dict(config.items(section))

    copy(args.config_file, args.log_dir)
    copy(args.graph_file, args.log_dir)
    utils.write_args(args, args.log_dir)

    g = Graph()
    g.read_graph_from_file(args.graph_file, args.graph_type)
    n_machines = args.machines
    procs_per_machine = args.procs_per_machine[0]

    l = Linear(n_machines, procs_per_machine)
    m_id = args.machine_id

    logging.basicConfig(level=logging.INFO)

    processes = []
    for r in range(procs_per_machine):
        processes.append(
            mp.Process(
                target=EL_Local,
                args=[
                    r,
                    m_id,
                    l,
                    g,
                    my_config,
                    args.iterations,
                    args.log_dir,
                    args.weights_store_dir,
                    log_level[args.log_level],
                    args.test_after,
                    args.train_evaluate_after,
                    args.reset_optimizer,
                    args.local_only,
                ],
            )
        )

    for p in processes:
        p.start()

    for p in processes:
        p.join()
