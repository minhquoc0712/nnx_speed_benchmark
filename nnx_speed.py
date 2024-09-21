import time
from functools import partial

import numpy as np

import torch
import jax
import jax.numpy as jnp
from flax import nnx
from flax import linen as nn


# ---------------------------
# PyTorch Version of the Model
# ---------------------------
class CNNPyTorch(torch.nn.Module):
    """A simple CNN model in PyTorch."""

    def __init__(self):
        super(CNNPyTorch, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), padding="same")
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same")
        self.pool = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = torch.nn.Linear(3136, 256)
        self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------
# NNX Version of the Model
# ---------------------------
class CNNNNX(nnx.Module):
    """A simple CNN model in NNX."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# ---------------------------
# Linen Version of the Model
# ---------------------------
class CNNLinen(nn.Module):
    """A simple CNN model in Linen."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


# ---------------------------
# Benchmarking Setup
# ---------------------------
def benchmark(fn, input_data, num_warmup=10, num_runs=1000, cooldown_time=3):
    # Warm-up phase
    for _ in range(num_warmup):
        _ = fn(input_data)
    time.sleep(cooldown_time)

    # Collect individual run times
    run_times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = fn(input_data)
        run_times.append(time.time() - start_time)

    time.sleep(cooldown_time)

    # Calculate mean and standard deviation of the run times
    mean_time = np.mean(run_times)
    std_time = np.std(run_times)

    return mean_time, std_time


input_shape = (64, 28, 28, 1)
dummy_input_jax = jnp.ones(input_shape, dtype=jnp.float32)
dummy_input_torch = torch.ones((32, 1, 28, 28), dtype=torch.float32)

# Instantiate models
rng = jax.random.PRNGKey(0)
model_nnx = CNNNNX(rngs=nnx.Rngs(rng))
model_linen = CNNLinen()
model_pytorch = CNNPyTorch()

# Initialize parameters for Linen
params_linen = model_linen.init(rng, dummy_input_jax)

# JIT compiled functions (NNX and Linen)
model_nnx_jit = nnx.jit(model_nnx)
model_linen_jit = jax.jit(lambda x: model_linen.apply(params_linen, x))

# TorchScript compiled function (PyTorch)
model_pytorch_scripted = torch.jit.script(model_pytorch)

# ---------------------------
# Test on CPU and GPU
# ---------------------------
for device in ["cpu", "gpu"]:
    print(f"Running on {device.upper()}...")

    # Switch to the specified device for JAX (CPU or GPU)
    input_data_jax = jax.device_put(dummy_input_jax, device=jax.devices(device)[0])

    if device == "cpu":
        # PyTorch CPU device
        model_pytorch.to("cpu")
        input_data_torch = dummy_input_torch
    else:
        # PyTorch GPU device
        model_pytorch.to("cuda")
        input_data_torch = dummy_input_torch.to("cuda")

    # Without JIT (NNX and Linen)
    nnx_time = benchmark(model_nnx, input_data_jax)
    linen_time = benchmark(model_linen_jit, input_data_jax)

    # With JIT (NNX and Linen)
    nnx_jit_time = benchmark(model_nnx_jit, input_data_jax)
    linen_jit_time = benchmark(model_linen_jit, input_data_jax)

    # PyTorch (without TorchScript)
    torch_time = benchmark(model_pytorch.forward, input_data_torch)

    # PyTorch (with TorchScript)
    torch_scripted_time = benchmark(model_pytorch_scripted.forward, input_data_torch)

    results = [
        ("NNX (No JIT)", nnx_time),
        ("Linen (No JIT)", linen_time),
        ("NNX (With JIT)", nnx_jit_time),
        ("Linen (With JIT)", linen_jit_time),
        ("PyTorch (No TorchScript)", torch_time),
        ("PyTorch (With TorchScript)", torch_scripted_time),
    ]

    use_sort = True
    if use_sort:
        # Sort the results by mean execution time (the first element in the tuple)
        results = sorted(results, key=lambda x: x[1][0])  # Sort by mean time (x[1][0])

    # Print the sorted results
    for model_name, (mean_time, std_time) in results:
        print(
            f"{model_name}: Mean = {mean_time:.4f} seconds, Std = {std_time:.4f} seconds"
        )
