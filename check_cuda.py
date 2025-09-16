import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory allocated:", torch.cuda.memory_allocated(0)/1024**2, "MB")
    print("GPU memory reserved:", torch.cuda.memory_reserved(0)/1024**2, "MB")
