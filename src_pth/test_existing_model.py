import torch as th
import torchvision as thv
import numpy as np
from PIL import Image
from model_def import RImNN
import sys

if __name__=="__main__":
  exNum = int(sys.argv[1])

  device = "cuda" if th.cuda.is_available() else "cpu"
  print(f"device = {device}")

  width  = 8
  height = 8
  channel_count = 16 # RGB, A - indicate if cell alive, .. 12 <= space for nn information

  model = RImNN(width, height, channel_count, 1)
  model.load_state_dict(th.load(f"../Models/ex{exNum}Model.pth"))
  model.eval()

  @th.no_grad()
  def rearrange_to_img(X):
    X = X[:, 0:4, ...]
    X = th.squeeze(X)
    X = th.moveaxis(X, 0, 2)
    X = X.numpy()
    X *= 255.
    X = X.astype(np.uint8)

    return X

  @th.no_grad()
  def save_img(X, name, in_right_format=False):
    if(in_right_format):  
       Image.fromarray(X).save(f"{name}.png")
    else:
      Image.fromarray(rearrange_to_img(X)).save(f"{name}.png")

  seed = th.zeros(1, channel_count, width, height)
  seed[:, 3:, width//2, height//2] = 1.0
  atlas = np.zeros(shape=(width*19, height*19, 4), dtype=np.uint8)

  for y in range(0, 19):
    for x in range(0, 19):
      seed = model(seed)
      atlas[x*8:(x+1)*8, y*8:(y+1)*8, :] = rearrange_to_img(seed.clone().detach())

  
  save_img(np.flip(np.rot90(atlas), axis=0), f"testEx{exNum}", in_right_format=True)


