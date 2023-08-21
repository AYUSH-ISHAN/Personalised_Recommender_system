import argparse
import os
import random
from collections import OrderedDict
from datasets.datasets import make
import torch.optim as optim
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import datasets
import models
import utils
import utils.optimizers as optimizers

import alg_parametes

def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False

  ckpt_name = args.name
  if ckpt_name is None:
    ckpt_name = config.encoder
    ckpt_name += '_' + config.dataset.replace('meta-', '')
    ckpt_name += '_{}_way_{}_shot'.format(
      config.train.n_way, config.train.n_shot)
  if args.tag is not None:
    ckpt_name += '_' + args.tag

  ckpt_path = os.path.join('./save', ckpt_name)
  utils.ensure_path(ckpt_path)
  utils.set_log_path(ckpt_path)
  writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
  yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

  ##### Dataset #####
  print(type(config.train))
  print(config)
  # meta-train
  set_X, set_Y, test_set_X, test_set_Y = datasets.make(config.dataset)
  # print(len(set_X))
  train_set_X = torch.from_numpy(set_X)
  train_set_Y = torch.from_numpy(set_Y).to(torch.long)
  test_set_X = torch.from_numpy(test_set_X)
  test_set_Y = torch.from_numpy(test_set_Y).to(torch.long)
  
  model = models.make(alg_parametes.classifier, alg_parametes.D_MODEL, alg_parametes.D_HIDDEN, alg_parametes.N_LAYERS, alg_parametes.N_HEAD, alg_parametes.D_K, alg_parametes.D_V, alg_parametes.N_POSITION)
  optimizer = optim.Adam(model.parameters(), lr=1e-5)
  start_epoch = 1
  max_va = 0.

  if args.efficient:
    model.go_efficient()

  if alg_parametes.PARALLEL:
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.compute_n_params(model)))
  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

  ##### Training and evaluation #####
    
  # 'tl': meta-train loss
  # 'ta': meta-train accuracy
  # 'vl': meta-val loss
  # 'va': meta-val accuracy
  aves_keys = ['tl', 'ta', 'vl', 'va']
  trlog = dict()
  for k in aves_keys:
    trlog[k] = []

  for epoch in range(start_epoch, alg_parametes.epoch + 1):
    timer_epoch.start()
    aves = {k: utils.AverageMeter() for k in aves_keys}
    inds = np.arange(train_set_X.shape[0])
    # print(inds)
    np.random.shuffle(inds)
    # meta-train
    model.train()
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    np.random.seed(epoch)

    for start in range(0, train_set_X.shape[0], alg_parametes.MINIBATCH_SIZE):
      end = start + alg_parametes.MINIBATCH_SIZE
      logits = model(train_set_X[start:end],train_set_X[start:end], train_set_Y[start:end], alg_parametes.inner_args, meta_train = True)
      # logits = logits.flatten(0, 1)
      if logits.shape[0] != 128:
        break
      logits = torch.reshape(logits, (128,11))
      labels = train_set_Y[start:end].flatten().to(torch.long)
      
      pred = torch.argmax(logits, dim=-1)
      print(pred)
      acc = utils.compute_acc(pred, labels)
      # print(logits.shape, labels.shape)
      loss = F.cross_entropy(logits, labels)
      print("train : ", acc, loss)
      aves['tl'].update(loss.item(), 1)
      aves['ta'].update(acc, 1)
      
      optimizer.zero_grad()
      loss.backward()
      for param in optimizer.param_groups[0]['params']:
        nn.utils.clip_grad_value_(param, 10)
      optimizer.step()

      # break

      # break
    # meta-val
    if alg_parametes.eval_val:
      model.eval()
      np.random.seed(0)
      for start in range(0, test_set_X.shape[0], alg_parametes.MINIBATCH_SIZE):
        end = start + alg_parametes.MINIBATCH_SIZE
        # logits = model(train_set_X[start:end],train_set_X[start:end], train_set_Y[start:end], alg_parametes.inner_args, meta_train = True)

        # x_shot, x_query, y_shot, y_query = data
        # x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
        # x_query, y_query = x_query.cuda(), y_query.cuda()

        if alg_parametes.inner_args.reset_classifier:
          if alg_parametes.PARALLEL:
            model.module.reset_classifier()
          else:
            model.reset_classifier()

        logits = model(test_set_X[start:end],test_set_X[start:end], test_set_Y[start:end], alg_parametes.inner_args, meta_train = True)
        # logits = logits.flatten(0, 1)
        if logits.shape[0] != 128:
          break  # the limit reached
        # print(logits.shape)
        logits = torch.reshape(logits, (128,11))
        # labels = y_query.flatten()
        labels = test_set_Y[start:end].flatten().to(torch.long)
        
        pred = torch.argmax(logits, dim=-1)
        acc = utils.compute_acc(pred, labels)
        loss = F.cross_entropy(logits, labels)
        aves['vl'].update(loss.item(), 1)
        aves['va'].update(acc, 1)

        # print(acc, loss)
        # break

    
    lr_scheduler = None # just for now
    
    if lr_scheduler is not None:
      lr_scheduler.step()

    for k, avg in aves.items():
      aves[k] = avg.item()
      trlog[k].append(aves[k])

    t_epoch = utils.time_str(timer_epoch.end())
    t_elapsed = utils.time_str(timer_elapsed.end())
    t_estimate = utils.time_str(timer_elapsed.end() / 
      (epoch - start_epoch + 1) * (alg_parametes.epoch - start_epoch + 1))

    # formats output
    log_str = 'epoch {}, meta-train {:.4f}|{:.4f}'.format(
      str(epoch), aves['tl'], aves['ta'])
    writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)
    writer.add_scalars('acc', {'meta-train': aves['ta']}, epoch)

    # if eval_val:
    #   log_str += ', meta-val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
    #   writer.add_scalars('loss', {'meta-val': aves['vl']}, epoch)
    #   writer.add_scalars('acc', {'meta-val': aves['va']}, epoch)

    log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
    utils.log(log_str)

    # saves model and meta-data
    # if config.get('_parallel'):
    if alg_parametes.PARALLEL:
      model_ = model.module
    else:
      model_ = model

    training = {
      'epoch': epoch,
      'max_va': max(max_va, aves['va']),

      'optimizer': alg_parametes.optimizer,
      'optimizer_args': alg_parametes.optimizer_args,
      'optimizer_state_dict': optimizer.state_dict(),
      'lr_scheduler_state_dict': lr_scheduler.state_dict() 
        if lr_scheduler is not None else None,
    }
    ckpt = {
      # 'file': __file__,
      # 'config': alg_parametes,
      # 'model' : model.state_dict()
      'encoder': alg_parametes.encoder,
      'encoder_args': alg_parametes.encoder_args,
      'encoder_state_dict': model_.encoder.state_dict(),

      'classifier': alg_parametes.classifier,
      # 'classifier_args': alg_parametes.classifier_args,
      'classifier_state_dict': model_.classifier.state_dict(),

      'training': training,
    }

    # 'epoch-last.pth': saved at the latest epoch
    # 'max-va.pth': saved when validation accuracy is at its maximum
    torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
    torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))

    if aves['va'] > max_va:
      max_va = aves['va']
      torch.save(ckpt, os.path.join(ckpt_path, 'max-va.pth'))

    writer.flush()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--name', 
                      help='model name', 
                      type=str, default=None)
  parser.add_argument('--tag', 
                      help='auxiliary information', 
                      type=str, default=None)
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  args = parser.parse_args()
  config = alg_parametes #yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  utils.set_gpu(args.gpu)
  main(config)