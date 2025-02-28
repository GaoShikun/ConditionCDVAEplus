"""
MIT License

Copyright (c) 2023 Yi-Lun Liao
"""


import logging

import hydra
import torch
import torch_geometric
from omegaconf import DictConfig
from torch_geometric.data import DataLoader

from ccdvaeplus.common.data_utils import frac_to_cart_coords
from ccdvaeplus.common.utils import PROJECT_ROOT
from ccdvaeplus.pl_data.dataset import CrystDataset

from ccdvaeplus.common.data_utils import radius_graph_pbc

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FileLogger:
    def __init__(self, is_master=False, is_rank0=False, output_dir=None, logger_name='training'):
        # only call by master
        # checked outside the class
        self.output_dir = output_dir
        if is_rank0:
            self.logger_name = logger_name
            self.logger = self.get_logger(output_dir, log_to_file=is_master)
        else:
            self.logger_name = None
            self.logger = NoOp()

    def get_logger(self, output_dir, log_to_file):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        if output_dir and log_to_file:
            time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
            debuglog = logging.FileHandler(output_dir + '/debug.log')
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)
            logger.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)

        # Reference: https://stackoverflow.com/questions/21127360/python-2-7-log-displayed-twice-when-logging-module-is-used-in-two-python-scri
        logger.propagate = False

        return logger

    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)

    def info(self, *args):
        self.logger.info(*args)


# no_op method/object that accept every signature
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op


def compute_stats(data_loader, logger, print_freq=1000):
    '''
        Compute mean of numbers of nodes and edges
    '''
    log_str = '\nCalculating statistics with '
    log_str = log_str + 'max_radius={}\n'.format(10.0)
    logger.info(log_str)


    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()

    for step, data in enumerate(data_loader):

        cart_coords = frac_to_cart_coords(
            data.frac_coords, data.lengths, data.angles, data.num_atoms)

        edge_index, to_jimages, num_bonds = radius_graph_pbc(
            cart_coords, data.lengths, data.angles, data.num_atoms, 6.0, 20,
            device=data.num_atoms.device)

        batch = torch.arange(data.num_atoms.size(0),
                             device=data.num_atoms.device).repeat_interleave(
            data.num_atoms, dim=0)        # edge_src, edge_dst = radius_graph(pos, r=max_radius, batch=batch,
        #                                   max_num_neighbors=1000)
        edge_src, edge_dst = edge_index
        batch_size = float(batch.max() + 1)
        num_nodes = cart_coords.shape[0]
        num_edges = edge_src.shape[0]
        num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
        num_degree = torch.sum(num_degree)

        avg_node.update(num_nodes / batch_size, batch_size)
        avg_edge.update(num_edges / batch_size, batch_size)
        avg_degree.update(num_degree / (num_nodes), num_nodes)

        if step % print_freq == 0 or step == (len(data_loader) - 1):
            log_str = '[{}/{}]\tavg node: {}, '.format(step, len(data_loader), avg_node.avg)
            log_str += 'avg edge: {}, '.format(avg_edge.avg)
            log_str += 'avg degree: {}, '.format(avg_degree.avg)
            logger.info(log_str)

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: DictConfig):

    from ccdvaeplus.common.data_utils import get_scaler_from_data_list
    _log = FileLogger(is_master=True, is_rank0=True, output_dir="E:\\Users\\PycharmProjects\\cdvae-copy")

    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    data_loader = DataLoader(data_list, batch_size=64, shuffle=True)

    compute_stats(data_loader, _log)

if __name__ == "__main__":
    main()