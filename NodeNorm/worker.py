import os
from typing import Callable

import wandb
import torch
from torch.nn.utils import clip_grad_norm_

from models import MultiOutGNN, GNN
from utils import accuracy, adjust_learning_rate
import numpy as np
from tqdm import tqdm
from copy import deepcopy as dcopy


def get_l1_regularization(model):
    regularization_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            regularization_loss += torch.sum(abs(param))
    return regularization_loss


def get_grads(module, param):
    sum_numel = np.zeros(2)
    if hasattr(module, param) \
       and getattr(module, param) is not None \
       and getattr(module, param).requires_grad:
        sum_numel[0] = getattr(module, param).grad.abs().sum().detach().cpu().numpy()
        sum_numel[1] = getattr(module, param).grad.numel()
    else:
        for each in module.named_children():
            name = each[0]
            sub_module = getattr(module, name)
            sum_numel += get_grads(sub_module, param)
    return sum_numel


def record_grads(w_grads, b_grads, model, epoch_idx, bias):
    grad_count = 0
    for layer in model.structure:
        layer_name = layer._get_name()
        grad_sum_numel = get_grads(layer, 'weight')
        if grad_sum_numel[1] != 0 and (layer_name.find('Batch') == -1):
            w_grad = grad_sum_numel[0] / grad_sum_numel[1]
            w_grads[grad_count][epoch_idx] = w_grad
            
            if bias:
                grad_sum_numel = get_grads(layer, 'bias')
                assert grad_sum_numel[1] != 1
                b_grad = grad_sum_numel[0] / grad_sum_numel[1]
                b_grads[grad_count][epoch_idx] = b_grad
            grad_count += 1


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm


def get_clip_value(gn_container, ref_window):
    if ref_window == -1:
        value = sum(gn_container) / len(gn_container)
    else:
        assert len(gn_container) >= ref_window
        value = sum(gn_container[-ref_window : ]) / ref_window
    return value


def test(criterion, data, pred_dict, model, model_state):
    features, labels, idx_train, idx_val, idx_test = data[1], data[2], data[3], data[4], data[5]

    ## note that also need to save predicts of train / val with last model
    with torch.no_grad():
        #model.load_state_dict(torch.load(model_path))
        model.load_state_dict(model_state)
        model.eval()
        output, node_emb = model(features, data[0])

        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_test, correct_test = accuracy(output[idx_test], labels[idx_test])

        print(f'loss= {loss_test.item(): .4f}\t'
              f'accuracy= {acc_test.item(): .4f}')

        pred_dict['test score'] = output[idx_test].cpu().detach().numpy()
        pred_dict['test correct'] = correct_test.cpu().detach().numpy()
        pred_dict['test acc'] = acc_test.cpu().detach().numpy()
        
    return pred_dict


def set_pred_dict(correct_train, correct_val, idx_train, idx_val, output):
    pred_dict = {}
    pred_dict['train score'] = output[idx_train].cpu().detach().numpy()
    pred_dict['train correct'] = correct_train.cpu().detach().numpy()
    pred_dict['val score'] = output[idx_val].cpu().detach().numpy()
    pred_dict['val correct'] = correct_val.cpu().detach().numpy()
    return pred_dict


def save_model(model, file_name, log_wandb=False, dir='models', state_dict=False):
    file_name = os.path.join(wandb.run.dir, file_name) if log_wandb else os.path.join(dir, file_name)
    torch.save(model if state_dict else model.state_dict(), file_name)
    if log_wandb:
        wandb.save(file_name, policy='now')


def train(config, criterion, data, model, optimizer, forward_params, group=None, name=None):
    run = wandb.init(project="gnn-depth", entity="gnn-depth", config=config, group=group, name=name)
    features, labels, idx_train, idx_val, idx_test = data[1], data[2], data[3], data[4], data[5]

    ## gradient
    gn_container = []
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []


    best_acc = 0
    no_improve_epochs = 0
    for epoch in tqdm(range(config['optim']['epoch']), bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
        adjust_learning_rate(optimizer=optimizer,
                             epoch=epoch,
                             lr_down_epoch_list=config['optim']['down_list'])
        model.train()
        adj = data[0]
        output, node_emb = model(x=features, graph=adj, **forward_params)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train, correct_train = accuracy(output[idx_train], labels[idx_train])
        optimizer.zero_grad()
        l1_regularization_loss = get_l1_regularization(model)
        loss_train += config['optim']['l1_weight'] * l1_regularization_loss
        loss_train.backward()
        if config['optim']['gnclip']['use'] and epoch > max(config['optim']['gnclip']['ref_window'], 0):
            clip_value = get_clip_value(gn_container, config['optim']['gnclip']['ref_window'])
            clip_grad_norm_(model.parameters(), clip_value, norm_type=config['optim']['gnclip']['norm'])

        clipped_grad_norm = get_grad_norm(model.parameters(), norm_type=config['optim']['gnclip']['norm'])
        gn_container.append(clipped_grad_norm)
        gn_container = gn_container[:config['optim']['gnclip']['ref_window']]

        optimizer.step()
        train_losses.append(loss_train.item())
        train_accs.append(acc_train.item())
        train_losses, train_accs = train_losses[-config['logging']['moving_avg']:], \
                                   train_accs[-config['logging']['moving_avg']:]

        # log
        wandb.log({'train loss': np.mean(train_losses), 'train acc': np.mean(train_accs)}, step=epoch + 1)

        if (epoch+1) % config['logging']['eval_freq'] == 0:
            model.eval()
            with torch.no_grad():
                output, _ = model(features, adj, **forward_params)

            loss_test = criterion(output[idx_test], labels[idx_test])
            acc_test, _ = accuracy(output[idx_test], labels[idx_test])
            test_losses.append(loss_test.item())
            test_accs.append(acc_test.item())
            test_losses, test_accs = test_losses[-config['logging']['moving_avg']:], \
                                       test_accs[-config['logging']['moving_avg']:]

            wandb.log({'test loss': np.mean(test_losses), 'test acc': np.mean(test_accs)}, step=epoch + 1)

            loss_val = criterion(output[idx_val], labels[idx_val])
            acc_val, _ = accuracy(output[idx_val], labels[idx_val])
            val_losses.append(loss_val.item())
            val_accs.append(acc_val.item())

            # check for early stopping
            if (epoch > config['optim']['min_epochs']) and (loss_val > np.mean(val_losses[:-1])):
                no_improve_epochs += 1
            else:
                no_improve_epochs = 0

            val_losses, val_accs = val_losses[-config['logging']['moving_avg']:], \
                                   val_accs[-config['logging']['moving_avg']:]
            wandb.log({'validation loss': np.mean(val_losses), 'validation acc': np.mean(val_accs)}, step=epoch + 1)

            if no_improve_epochs >= config['optim']['patience']:
                break

            # saving best model
            if acc_val > best_acc:
                best_acc = acc_val
                best_model = dcopy(model.state_dict())

        if (epoch+1) % config['logging']['checkpoint_freq'] == 0:
            save_model(model, f'model_{epoch+1}', log_wandb=True)
    save_model(model, 'model', log_wandb=True)
    save_model(best_model, 'best_model', log_wandb=True, state_dict=True)
    run.finish()

    return model


def train_per_depth_freeze(config, criterion: Callable, data, model: GNN, optimizer, layers):
    """
    :param config: dictionary of params
    :param criterion: loss function
    :param data: dataset
    :param model: nn.Module
    :param optimizer: -
    :param layers: the layers we'll use as an output point of the model

    """
    for freeze_layer, out_layer in list(
            zip([0] + layers[:-1], layers)):

        forward_params = {}
        if out_layer is not None:
            forward_params = {"out_layer": out_layer}

        for p_name, param in model.named_parameters():
            if (freeze_layer == 0) or (int(p_name.split('.')[1].split('_')[0]) == freeze_layer-1):
                break
            param.requires_grad = False

        model = train(config, criterion, data, model, optimizer, forward_params, group='baseline_freeze',
                      name=f"depth={out_layer}")

        for p_name, param in model.named_parameters():
            if (freeze_layer == 0) or (int(p_name.split('.')[1].split('_')[0]) == freeze_layer-1):
                break
            param.requires_grad = True

    return model


def train_multiout_naive(config, criterion, data, model, optimizer, layers):
    run = wandb.init(project="gnn-depth", entity="gnn-depth", config=config, group='baseline_naive')
    features, labels, idx_train, idx_val, idx_test = data[1], data[2], data[3], data[4], data[5]

    ## gradient
    gn_container = []
    train_losses, train_accs = [], [[] for _ in layers]
    val_losses, val_accs = [], [[] for _ in layers]
    test_losses, test_accs = [], [[] for _ in layers]

    no_improve_epochs = 0
    for epoch in tqdm(range(config['optim']['epoch']), bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
        model.train()
        adj = data[0]
        output, _ = model(x=features, graph=adj, layers=layers)
        loss_train = sum([criterion(o[idx_train], labels[idx_train]) for o in output])

        acc_train = [accuracy(o[idx_train], labels[idx_train])[0] for o in output]
        optimizer.zero_grad()
        l1_regularization_loss = get_l1_regularization(model)
        loss_train += config['optim']['l1_weight'] * l1_regularization_loss
        loss_train.backward()
        if config['optim']['gnclip']['use'] and epoch > max(config['optim']['gnclip']['ref_window'], 0):
            clip_value = get_clip_value(gn_container, config['optim']['gnclip']['ref_window'])
            clip_grad_norm_(model.parameters(), clip_value, norm_type=config['optim']['gnclip']['norm'])

        clipped_grad_norm = get_grad_norm(model.parameters(), norm_type=config['optim']['gnclip']['norm'])
        gn_container.append(clipped_grad_norm)
        gn_container = gn_container[:config['optim']['gnclip']['ref_window']]

        optimizer.step()
        train_losses.append(loss_train.item())
        train_losses = train_losses[-config['logging']['moving_avg']:]
        wandb.log({'train loss': np.mean(train_losses)}, step=epoch + 1)
        for i in range(len(train_accs)):
            train_accs[i].append(acc_train[i].item())
            train_accs[i] = train_accs[i][-config['logging']['moving_avg']:]
            wandb.log({f'train acc d={layers[i]}': np.mean(train_accs[i])}, step=epoch + 1)

        if (epoch + 1) % config['logging']['eval_freq'] == 0:
            model.eval()
            with torch.no_grad():
                output, _ = model(features, adj, layers=layers)

            loss_test = sum([criterion(o[idx_test], labels[idx_test]) for o in output])
            acc_test = [accuracy(o[idx_test], labels[idx_test])[0] for o in output]
            test_losses.append(loss_test.item())
            test_losses = test_losses[-config['logging']['moving_avg']:]
            wandb.log({'test loss': np.mean(test_losses)}, step=epoch + 1)
            for i in range(len(test_accs)):
                test_accs[i].append(acc_test[i].item())
                test_accs[i] = test_accs[i][-config['logging']['moving_avg']:]
                wandb.log({f'test acc d={layers[i]}': np.mean(test_accs[i])}, step=epoch + 1)

            loss_val = sum([criterion(o[idx_val], labels[idx_val]) for o in output])
            acc_val = [accuracy(o[idx_val], labels[idx_val])[0] for o in output]

            # check for early stopping
            if (epoch > config['optim']['min_epochs']) and (loss_val > np.mean(val_losses[:-1])):
                no_improve_epochs += 1
            else:
                no_improve_epochs = 0

            val_losses.append(loss_val.item())
            val_losses = val_losses[-config['logging']['moving_avg']:]
            wandb.log({'validation loss': np.mean(val_losses)}, step=epoch + 1)
            for i in range(len(val_accs)):
                val_accs[i].append(acc_val[i].item())
                val_accs[i] = val_accs[i][-config['logging']['moving_avg']:]
                wandb.log({f'validation acc d={layers[i]}': np.mean(val_accs[i])}, step=epoch + 1)

            if no_improve_epochs >= config['optim']['patience']:
                break

        if (epoch + 1) % config['logging']['checkpoint_freq'] == 0:
            save_model(model, f'model_{epoch + 1}', log_wandb=True)
    save_model(model, 'model', log_wandb=True)
    run.finish()

    return model
