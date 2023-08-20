
dataset= 'meta-mini-imagenet'
eval_val = True
PARALLEL = True
class train:
  split= 'meta-train'
  image_size= 84
  normalization= False
  transform= None
  n_batch= 200
  n_episode= 4
  n_way= 5
  n_shot= 5
  n_query= 15
class val:
  split='meta-val'
  image_size= 84
  normalization= False
  transform= None
  n_batch= 200
  n_episode= 4
  n_way= 5
  n_shot= 5
  n_query= 15
get = 'val'
encoder= 'transformer'
class encoder_args:
  class bn_args:
    track_running_stats= False
classifier= 'logistic'

class inner_args:
  n_step= 5
  encoder_lr= 0.01
  classifier_lr= 0.01
  first_order= False
  reset_classifier= True
  weight_decay= 5.e-4
  frozen='- bn'
  momentum= 0.9

optimizer= 'adam'
class optimizer_args:
  lr= 0.001

epoch= 300

NET_SIZE = 14#512
D_MODEL = NET_SIZE  # for input and inner feature of attention
D_HIDDEN = 1024  # for feed-forward network
N_LAYERS = 1  # number of computation block
N_HEAD = 8
D_K = 32
D_V = 32
N_POSITION = 128
MINIBATCH_SIZE = 128