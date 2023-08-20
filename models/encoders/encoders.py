import torch
from models.encoders.transformer.encoder_model import TransformerEncoder
models = {}

def register(name):
  def decorator(cls):
    models[name] = cls
    return cls
  return decorator


def make(d_model, d_hidden, n_layers, n_head, d_k, d_v, n_position):
  # if name is None:
  #   return None
  model = TransformerEncoder(d_model, d_hidden, n_layers, n_head, d_k, d_v, n_position)
  # model = models[name](**kwargs)
  if torch.cuda.is_available():
    model.cuda()
  return model


def load(ckpt):
  model = make(ckpt['encoder'], **ckpt['encoder_args'])
  if model is not None:
    model.load_state_dict(ckpt['encoder_state_dict'])
  return model