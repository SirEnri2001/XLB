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
        # x = nn.BatchNorm(use_running_average=True)(x)
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
    def __call__(self, x, left_part):
        x = nn.ConvTranspose(self.features, kernel_size=(2, 2), strides=self.up_factor, padding='SAME')(x)
        x = jax.lax.concatenate([left_part, x], dimension=3)
        x = ConvBlock(self.features)(x)
        x = ConvBlock(self.features)(x)
        return x


class UNet(nn.Module):
    """UNet architecture with contracting and expanding paths."""
    features_start: int = 64

    @nn.compact
    def __call__(self, x):
        # Contracting path
        down1 = DownBlock(self.features_start)(x)
        down1_max_pooled = nn.max_pool(down1, window_shape=(2, 2), strides=(2, 2))
        up1 = UpBlock(self.features_start)(down1_max_pooled, down1)
        output = nn.Dense(2)(up1)
        return output


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
  params = module.init(rng, jnp.ones([1, 110, 20, 2]))['params'] # initialize parameters by passing a template image
  tx = optax.sgd(learning_rate, momentum)
  return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
      metrics=Metrics.empty())

@partial(jax.jit)
def normalize_frame(frame):
    min = frame.min()
    max = frame.max()
    return (frame - min) / (max - min)

@partial(jax.vmap, in_axes=(0), out_axes=0)
def vmapped_normalize_frame(frame):
    return normalize_frame(frame)

@partial(jax.jit, static_argnums=(2))
def train_step(state, batch_data, low_res_lbm_solver):
    """Train for a single step."""
    def loss_fn(params):
        batched_f = batch_data['f_poststreaming']
        _, high_res_u = low_res_lbm_solver.vmapped_update_macroscopic(batched_f)
        high_res_u = vmapped_normalize_frame(high_res_u)
        low_res_step_output = low_res_lbm_solver.vmapped_run_step(0, batched_f)
        correction = state.apply_fn({'params': params}, low_res_step_output['u'][0])
        loss = optax.l2_loss(normalize_frame(low_res_step_output['u'][0]) + 0.01 * correction, high_res_u).sum()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    train_loss = loss_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, train_loss

@jax.jit
def pred_step(state, batch):
  return state.apply_fn({'params': state.params}, batch)