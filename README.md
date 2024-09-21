# nnx_speeed_benchmark

Different runs could produce different results.

```
2024-09-21 13:54:00.039647: W external/xla/xla/service/gpu/nvptx_compiler.cc:893] The NVIDIA driver's CUDA version is 12.3 which is older than the PTX compiler version 12.5.82. Because the driver is older than the PTX compiler version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
Running on CPU...
PyTorch (With TorchScript): Mean = 0.0040 seconds, Std = 0.0003 seconds
PyTorch (No TorchScript): Mean = 0.0043 seconds, Std = 0.0006 seconds
Linen (With JIT): Mean = 0.0071 seconds, Std = 0.0014 seconds
Linen (No JIT): Mean = 0.0072 seconds, Std = 0.0014 seconds
NNX (With JIT): Mean = 0.0075 seconds, Std = 0.0016 seconds
NNX (No JIT): Mean = 0.0181 seconds, Std = 0.0023 seconds
Running on GPU...
NNX (With JIT): Mean = 0.0005 seconds, Std = 0.0004 seconds
Linen (With JIT): Mean = 0.0005 seconds, Std = 0.0005 seconds
Linen (No JIT): Mean = 0.0005 seconds, Std = 0.0004 seconds
PyTorch (No TorchScript): Mean = 0.0007 seconds, Std = 0.0001 seconds
PyTorch (With TorchScript): Mean = 0.0007 seconds, Std = 0.0001 seconds
NNX (No JIT): Mean = 0.0037 seconds, Std = 0.0008 seconds
```
