import torch as th
import torchvision as thv
from PIL import Image
import numpy as np
from model_def import RImNN

## EX DESC
# NOTE create pool of samples initialize grid with zeros except cell in the center
# NOTE cell's going to has all channels set to 1 except RGB
# NOTE sample specified number of batches from the pool
# NOTE apply update rule random number of times
# NOTE apply L2 loss for each pixel between output img and target img to regenerate
# NOTE replace one with the highest loss to freshly seeded sample
# NOTE as in RNNs differentiably optimize parameters 

if __name__=="__main__":

  #CONSTANTS 
  WIDTH  = 8
  HEIGHT = 8
  CHANNEL_COUNT = 16 # RGB, A - indicate if cell alive, .. 12 <- space for nn information
  POOL_SIZE = 256
  BATCH_SIZE = 8

  #LOADING TARGETS
  target_img = np.array(Image.open("../Images/place_1.png"))
  assert(tuple(target_img.shape) == (WIDTH, HEIGHT, 4))

  WIDTH, HEIGHT, CHANNEL_COUNT = target_img.shape
  assert(CHANNEL_COUNT == 4)

  CHANNEL_COUNT += 12

  target_img = np.moveaxis(target_img, 2, 0)
  target_img = target_img[None , ...]
  target_img = th.from_numpy(target_img)
  target_img = target_img.float() / 255.

  #CREATING POOL
  seed = th.zeros(CHANNEL_COUNT, WIDTH, HEIGHT)
  seed[3:, WIDTH//2, HEIGHT//2] = 1.0
  pool = th.repeat_interleave(seed[None, ...], POOL_SIZE, dim=0)
  pool_losses = th.rand(POOL_SIZE)
  
  #DEFINE MODEL LOSS AND OPTIMIZER 
  model = RImNN(WIDTH, HEIGHT, CHANNEL_COUNT, BATCH_SIZE)
  model.train()
  loss_fn = th.nn.MSELoss()
  optimizer = th.optim.Adam(model.parameters(), lr=2e-3) # 1e-3

  ##DEF. TRAIN STEP
  def train_step(X):
    num = int(np.random.uniform(POOL_SIZE, 96))
    optimizer.zero_grad()
    for _ in range(num):
      X = model(X)
    loss = loss_fn(X[:, 0:4, ...], target_img)
    loss.backward()#retain_graph=True)

    optimizer.step() 
    return X, loss

  #TRAINING
  def train():
    for _ in range(1800):
      indices = th.randint(0, POOL_SIZE, (BATCH_SIZE,))
      X = pool[indices, ...].clone().detach()
      X_losses = pool_losses[indices, ...].clone().detach()
      with th.no_grad():
        X_worst_index = th.argmax(X_losses)
        X[X_worst_index] = seed
      
      Y, loss = train_step(X)

      del X
      del X_losses

      pool[indices, ...] = Y
      pool_losses[indices, ...] = loss
      del indices

      print(f"{loss.item()}")

      #if(loss.item() <= 0.1):
      #  break

      del Y
      del loss

      

  train()
  th.save(model.state_dict(), "../Models/ex1Model.pth")
