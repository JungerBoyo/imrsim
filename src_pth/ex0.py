import torch as th
import torchvision as thv
from PIL import Image
import numpy as np
from model_def import RImNN

## EX DESC
# NOTE initialize grid with zeros except cell in the center
# NOTE cell's going to has all channels set to 1 except RGB
# NOTE apply update rule random number of times
# NOTE at last apply L2 loss for each pixel between output
# img and target img to regenerate
# NOTE as in RNNs differentiably optimize parameters of 
# update rule

if __name__=="__main__":
  width  = 8
  height = 8
  channel_count = 16 # RGB, A - indicate if cell alive, .. 4 <= space for nn information

  img = np.array(Image.open("../Images/place_1.png"))
  assert(tuple(img.shape) == (width, height, 4))

  img = np.moveaxis(img, 2, 0)
  img = img[None , ...]
  img = th.from_numpy(img)
  img = img.float() / 255.

  model = RImNN(width, height, channel_count)

  loss_fn = th.nn.MSELoss()
  optimizer = th.optim.Adam(model.parameters(), lr=2e-3) # 1e-3
  ## TRAIN 
  def train_step(X):
    num = int(np.random.uniform(64, 96))
    optimizer.zero_grad()
    for _ in range(num):
      X = model(X)
      loss = loss_fn(X[:, 0:4, ...], img)
      loss.backward(retain_graph=True)

    optimizer.step() 
    return X, loss
   
  BATCH_SIZE = 1

  # LOOP 
  for _ in range(1000):
    seed = th.zeros(BATCH_SIZE, channel_count, width, height)
    seed[:, 3:, width//2, height//2] = 1.0

    seed, loss = train_step(seed)

    print(f"{loss.item()}")

  th.save(model.state_dict(), "../Models/ex0Model.pth")