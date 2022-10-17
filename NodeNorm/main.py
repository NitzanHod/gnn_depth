from models import GNN, MultiOutGNN
from worker import train_per_depth_freeze, train_multiout_naive, train
from initializer import init_logger, init_seed, init_optimizer

from config import ex
import torch
import torch.nn as nn

from copy import deepcopy as dcopy
from utils import load_data, resplit


@ex.automain
def main(_run, _config, _log):
    """
    _config: dictionary; its keys and values are the variables setting in the cfg function
    _run: run object defined by Sacred, can be used to record hashable values and get some information
    _log: logger object provided by Sacred, but is not very flexible, we can define loggers by oureselves
    """
    config = dcopy(_config)
    torch.cuda.set_device(config['gpu_id'])

    data = load_data(config=config)
    split_iterator = range(config['data']['random_split']['num_splits']) \
        if config['data']['random_split']['use'] else range(1)

    config['adj'] = data[0]

    for i in split_iterator:
        if config['data']['random_split']['use']:
            data = resplit(dataset=config['data']['dataset'],
                           data=data,
                           full_sup=config['data']['full_sup'],
                           num_classes=torch.unique(data[2]).shape[0],
                           num_nodes=data[1].shape[0],
                           num_per_class=config['data']['label_per_class'],
                           )
            print(torch.sum(data[3]))

        model = MultiOutGNN(config=config)
        # model = GNN(config=config)

        if config['use_gpu']:
            model = model.cuda()
            data = [each.cuda() if hasattr(each, 'cuda') else each for each in data]
            data[0] = data[0].to(f"cuda:{config['gpu_id']}")
        optimizer = init_optimizer(params=model.parameters(), optim_type=config['optim']['type'],
                                   lr=config['optim']['lr'], weight_decay=config['optim']['weight_decay'],
                                   momentum=config['optim']['momemtum'])

        criterion = nn.NLLLoss()

        # option 1: train for each depth while freezing params
        # model = train_per_depth_freeze(config, criterion, data, model, optimizer, config['arch']['output_layers'])
        # option 2: train for all multi-out by summing the losses
        model = train_multiout_naive(config, criterion, data, model, optimizer, config['arch']['output_layers'])
        # option 3: train seperate model for each depth
        # model = train(config, criterion, data, model, optimizer, {}, group='baseline',
        #               name=f"depth={config['arch']['num_layers']}")

