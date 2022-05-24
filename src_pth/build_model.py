import torch as th
import torchvision as thv
from model_def import RImNN

if __name__=="__main__":
  device = "cuda" if th.cuda.is_available() else "cpu"
  print(f"device = {device}")

  # Tenors
  # NOTE tensor attribs { shape, dtype, device }
  # NOTE concat tensors with th.car([...], dim=...)
  # NOTE .item() converts to python numerical value
  # NOTE ops with _ postfix are inplace ops
  # NOTE convertsion to np using .numpy() returns ref same .from_numpy(n)
  # NOTE defining tensor with requires_grad=True makes it valid for AD
  # NOTE defer grad reqs is possible through .requires_grad_(True)

  # NNs
  # NOTE define layers and stuff in __init__
  # NOTE define call in forward
  # NOTE call with operator(X)
  # NOTE .Flatten i.e from 2D to 1D
  # NOTE .Linear, plain densly connected layer 
  # NOTE .Sequential, stack modules in order
  # NOTE call .named_parameters to analize layers' params

  width  = 32
  height = 32
  channel_count = 16 # RGB, A - indicate if cell alive, .. 12 <= space for nn information

  model = RImNN(width, height, channel_count)


