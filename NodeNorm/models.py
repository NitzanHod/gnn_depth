import torch.nn as nn
import torch.nn.functional as F
from layers import GNNBasicBlock, get_activation


class GNN(nn.Module):
    def __init__(self, config):
        super(GNN, self).__init__()
        self.arch = config['arch']
        self.n_layers = config['arch']['num_layers']
        self.structure = nn.Sequential()
        self.aggregator = {}
        self.num_classes = config['data']['num_classes']
        self.normalization = config['arch']['norm']
        self.construct_from_blocks()
        self.layer_names = [each[0] for each in list(self.structure.named_children())]
        self.batch_size = config['optim']['batch_size']

        assert not self.layer_names[-1].startswith('relu')

    def construct_from_blocks(self):
        l = 0
        for block in self.arch['structure']:
            layer_type = block[0]

            if layer_type != 'dropout':
                block_type, hyperparams = block[1], block[2]
            else:
                hyperparams = block[1]

            if layer_type in ['gcn', 'gcn_res', 'sage', 'sage_res']:
                in_channels, out_channels, activation, bias = \
                    hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3],

                self.structure.add_module(
                    f'{l}_{layer_type}', GNNBasicBlock(layer_type=layer_type,
                                                       block_type=block_type,
                                                       activation=('no', None),
                                                       normalization=self.normalization,
                                                       in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       bias=bias,
                                                       )
                )
                acti_type, acti_hyperparam = activation
                if acti_type != 'no':
                    self.structure.add_module(f'{l}_activation', get_activation(acti_type, acti_hyperparam))
                l+=1

            elif layer_type in ['gat', 'gat_res']:
                in_channels, out_channels, num_heads, activation, feat_drop, attn_drop = \
                    hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3], hyperparams[4], hyperparams[5]

                self.structure.add_module(
                    f'{l}_gat', GNNBasicBlock(layer_type='gat',
                                              block_type=block_type,
                                              activation=('no', None),
                                              normalization=self.normalization,
                                              in_channels=in_channels,
                                              out_channels=out_channels,
                                              num_heads=num_heads,
                                              feat_drop=feat_drop,
                                              attn_drop=attn_drop,
                                              )
                )
                acti_type, acti_hyperparam = activation
                if acti_type != 'no':
                    self.structure.add_module(f'{l}_{acti_type}', get_activation(acti_type, acti_hyperparam))
                l += 1
            elif layer_type == 'dropout':
                self.structure.add_module(
                     f'{l-1}_dropout', nn.Dropout(p=hyperparams[0])
                 )
            else:
                raise NotImplementedError

    def forward(self, x, graph):
        node_emb = []
        for l, block in enumerate(self.structure):
            name = self.layer_names[l].split('_')[1]
            if name not in ['dropout', 'activation']:
                x = block(graph, x)
                node_emb.append(x)
            else:
                x = block(x)
        return F.log_softmax(x, dim=1), node_emb


class MultiOutGNN(GNN):
    def __init__(self, config):
        super(MultiOutGNN, self).__init__(config)
        self.layers = config['arch']['output_layers']
        block = self.arch['structure'][-1]
        layer_type = block[0]
        block_type, hyperparams = block[1], block[2]
        in_channels, out_channels, activation, bias = \
                    hyperparams[0], hyperparams[1], hyperparams[2], hyperparams[3],

        self.out_heads = nn.ModuleList([GNNBasicBlock(layer_type=layer_type,
                       block_type=block_type,
                       activation=('no', None),
                       normalization=self.normalization,
                       in_channels=in_channels,
                       out_channels=out_channels,
                       bias=bias,
                       ) for _ in self.layers])

    def forward(self, x, graph, out_layer=None, layers=None):
        if (layers is None) and (out_layer is None):
            out_layer = self.n_layers
        outs = []
        for l, block in enumerate(self.structure):
            layer_num, name = self.layer_names[l].split('_')
            layer_num = int(layer_num)
            if name not in ['dropout', 'activation']:
                if (out_layer is not None) and layer_num+1 == out_layer:
                    idx = self.layers.index(layer_num+1)
                    outs = F.log_softmax(self.out_heads[idx](graph, x), dim=1)
                    break
                if (out_layer is None) and (layer_num+1 in layers):
                    idx = self.layers.index(layer_num+1)
                    outs.append(F.log_softmax(self.out_heads[idx](graph, x), dim=1))
                x = block(graph, x)
            else:
                x = block(x)
        return outs, None