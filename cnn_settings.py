from flax import linen as nn
import jax
import jax.numpy as jnp

class ConvBlock(nn.Module):
    """Defines a convolutional block with activation and normalization."""
    features: int
    kernel_size: int = (3, 3)
    strides: int = 1

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(self.features, kernel_size=self.kernel_size, strides=self.strides, padding='SAME')(inputs)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        return x


class DownBlock(nn.Module):
    """Downsamples feature maps through convolutions and pooling."""
    features: int
    pool_factor: int = 2

    @nn.compact
    def __call__(self, x):
        x = ConvBlock(self.features)(x)
        x = ConvBlock(self.features)(x)
        return x


class UpBlock(nn.Module):
    """Upsamples feature maps and concatenates with features from the contracting path."""
    features: int
    up_factor: int = 2

    @nn.compact
    def __call__(self, x):
        x = ConvBlock(self.features)(x)
        x = ConvBlock(self.features)(x)
        x = nn.ConvTranspose(self.features, kernel_size=(2, 2), strides=self.up_factor, padding='VALID')(x)
        return x


class UNet(nn.Module):
    """UNet architecture with contracting and expanding paths."""
    features_start: int = 64

    @nn.compact
    def __call__(self, x):
        input_shape = x.shape
        # Contracting path
        down1 = DownBlock(self.features_start * 2)(x)
        down1_max_pooled = nn.max_pool(down1, window_shape=(2, 2), strides=(2, 2))
        down2 = DownBlock(self.features_start * 4)(down1_max_pooled)
        down2_max_pooled = nn.max_pool(down2, window_shape=(2, 2), strides=(2, 2))
        down3 = DownBlock(self.features_start * 8)(down2_max_pooled)
        down3_max_pooled = nn.max_pool(down3, window_shape=(2, 2), strides=(2, 2))
        down4 = DownBlock(self.features_start * 16)(down3_max_pooled)
        down4_max_pooled = nn.max_pool(down4, window_shape=(2, 2), strides=(2, 2))

        # Expanding path with concatenation
        up1 = UpBlock(self.features_start * 16)(down4_max_pooled)
        down4_sliced = jax.lax.slice(down4, (4, 4, 0), (down4.shape[0] - 4, down4.shape[1] - 4, down4.shape[2]))
        up1_concatenated = jax.lax.concatenate([down4_sliced, up1], dimension=2)
        up2 = UpBlock(self.features_start * 4)(up1_concatenated)
        down3_sliced = jax.lax.slice(down3, (4, 4, 0), (down3.shape[0] - 4, down3.shape[1] - 4, down3.shape[2]))
        up2_concatenated = jax.lax.concatenate([down3_sliced, up2], dimension=2)
        up3 = UpBlock(self.features_start * 2)(up2_concatenated)
        print(up3.shape)
        return up3


class SimpleNet(nn.Module):
    features: int = 32
    kernel_size: int = (5, 5)
    strides: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=self.kernel_size, strides=self.strides, padding='SAME')(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(self.features, kernel_size=self.kernel_size, strides=self.strides, padding='SAME')(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(self.features, kernel_size=self.kernel_size, strides=self.strides, padding='SAME')(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.features)(x)
        x = nn.Conv(2, kernel_size=self.kernel_size, strides=self.strides, padding='SAME')(x)
        return x
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers
from functools import partial
@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')
class TrainState(train_state.TrainState):
  metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum):
  """Creates an initial `TrainState`."""
  params = module.init(rng, jnp.ones([1, 440, 82, 2]))['params'] # initialize parameters by passing a template image
  tx = optax.sgd(learning_rate, momentum)
  return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
      metrics=Metrics.empty())

@partial(jax.jit, static_argnums=(2, 3))
def train_step(state, ref_data, low_res_lbm_solver, frame_idx = 0):
  """Train for a single step."""
  def loss_fn(params, batch_data, selected_ts):
    _ , high_res_u = low_res_lbm_solver.update_macroscopic(batch_data['f_poststreaming'][selected_ts + 1])
    low_res_step_output = low_res_lbm_solver.run_step(selected_ts+1, selected_ts, batch_data['f_poststreaming'][selected_ts])
    correction = state.apply_fn({'params': params}, low_res_lbm_solver.saved_data['u'][0])
    loss = optax.l2_loss(low_res_step_output['u'][0]+correction, high_res_u).sum()
    return loss
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params, ref_data, frame_idx)
  state = state.apply_gradients(grads=grads)
  return state

@jax.jit
def pred_step(state, batch):
  return state.apply_fn({'params': state.params}, batch)