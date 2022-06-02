import torch as th
import torchvision as thv

###  CONDITIONS
# NOTE cells with A > 0.1 live, cells with A <= 0.1 but with 
# living neighbours live
# NOTE cells with A <= 0.1 without living neighbours 
# are considered dead, at each step explcitly set
# their values to 0.0

###  UPDATE RULE
## NOTE passed input X is of shape [w, h, chn]
## NOTE apply 3x3 convolution kernel to each cell
## PHASES
# NOTE 1. run 3x3 convolution with fixed weights/kernels
# sobel to retrieve gradient info and residual identity kernel 
# concat the gradients with identity which gives 
# [w, h, 3*chn output]
# NOTE 2. pass previous stage output to core sequential 
# submodel defined as 
'''self.core = th.nn.Sequential(
       th.nn.Conv2d(w*h, 128, 1),
       th.nn.ReLU(),
       th.nn.Conv2d(128, chn, 1)*
     )'''
# *weights initialized to zeros
# NOTE 3. simulate async update by stochastic mask which
# zeros out/drops out ouputted update from previous step
# NOTE 4. maskout dead cells based on CONDITIONS 


class RImNN(th.nn.Module):
  def __init__(self, width, height, channel_num, batch_size):
    super(RImNN, self).__init__()
    sobel_x = th.Tensor([[[1., 2., 1.],[0., 0., 0.],[-1.,-2.,-1.]]]) / 8.
    sobel_y = th.Tensor([[[1., 0.,-1.],[2., 0.,-2.],[ 1., 0.,-1.]]]) / 8. 
    identt  = th.Tensor([[[0., 0., 0.],[0., 1., 0.],[ 0., 0., 0.]]]) 
    self.kernels = th.stack([identt, sobel_x, sobel_y], 0)
    self.kernels = th.repeat_interleave(self.kernels, batch_size, 1)

    self.w = width
    self.h = height
    self.chn = channel_num
    self.bs = batch_size

    last_conv = th.nn.Conv2d(96, channel_num, 1)
    last_conv.weight.data.fill_(0.0)
    self.core = th.nn.Sequential(
      th.nn.Conv2d(3*channel_num, 96, 1),
      th.nn.ReLU(),
      last_conv
    )

  @th.no_grad() 
  def fixed_convolve(self, X): 
    X = th.reshape(X, shape=(self.chn, self.bs, self.w, self.h)) 
    out = th.nn.functional.conv2d(X, self.kernels, padding=1)
    out = out.reshape(1, 3*self.chn, self.w, self.h)
  
    return out

  @th.no_grad()
  def get_life_mask(self, X):
    life_mask = th.max_pool2d(X[:, 3:4, :, :], 3, 1, 1) > 0.1
    return life_mask

  def forward(self, X):
    # 1. 
    Y = self.fixed_convolve(X)

    # 2.
    dx = self.core(Y)

    # 3.
    X = X + dx * (th.rand_like(dx) < 0.5).float()

    # 4. 
    X = X * self.get_life_mask(X).float()

    return X 
