import os

import torch
import json
import pandas as pd
import numpy as np

DEFAULT_ROOT = './materials'
datasets = {}

def register(data):
  def decorator(cls):
    datasets[data] = cls
    return cls
  return decorator


# def make(data, **kwargs):
#   if kwargs.get('root_path') is None:
#     kwargs['root_path'] = os.path.join(DEFAULT_ROOT, data.replace('meta-', ''))
#   dataset = datasets[data](**kwargs)
#   return dataset

def make(folder_dir):
   df = pd.read_csv("./datasets/final_data.csv")
   features = df.iloc[:, :-1]
   target = df.iloc[:, -1]
   numpy_arrays_features = features.to_numpy(dtype=np.float32)
   numpy_arrays_target = target.to_numpy(dtype=np.float32)

   
   
   train_set_x = [numpy_arrays_features[i] for i in range(70000)]
   train_set_y = [numpy_arrays_target[i] for i in range(70000)]
   test_set_x = [numpy_arrays_features[i] for i in range(70000, len(numpy_arrays_target))]
   test_set_y = [numpy_arrays_target[i] for i in range(70000, len(numpy_arrays_target))]

   train_set_x, train_set_y, test_set_x, test_set_y = np.array(train_set_x), np.array(train_set_y),\
   np.array(test_set_x), np.array(test_set_y)
   
   
   return train_set_x, train_set_y, test_set_x, test_set_y



def collate_fn(batch):
  shot, query, shot_label, query_label = [], [], [], []
  for s, q, sl, ql in batch:
    shot.append(s)
    query.append(q)
    shot_label.append(sl)
    query_label.append(ql)
  
  shot = torch.stack(shot)                # [n_ep, n_way * n_shot, C, H, W]
  query = torch.stack(query)              # [n_ep, n_way * n_query, C, H, W]
  shot_label = torch.stack(shot_label)    # [n_ep, n_way * n_shot]
  query_label = torch.stack(query_label)  # [n_ep, n_way * n_query]
  
  return shot, query, shot_label, query_label
  
# def collate_fn(batch):
#   numpy_arrays_features, numpy_arrays_target = [] , []
#   for feature , target in batch :
#     numpy_arrays_features.append(feature)
#     numpy_arrays_target.append(target)

#   numpy_arrays_features = torch.stack(numpy_arrays_features)
#   numpy_arrays_target = torch.stack(numpy_arrays_target)
#   return numpy_arrays_features, numpy_arrays_target
