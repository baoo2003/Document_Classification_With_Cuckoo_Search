# Document_Classification_With_Cuckoo_Search

Xây dựng hệ thống phân loại văn bản tiếng Việt bằng thuật toán Cuckoo Search

**Environment & IDE**
Version: Python >= 3.11.9
IDE: VS Code

**How to run?**

1. Setting virtual env:

- Ctrl + Shift + N
- Click choose Python: Create environment ...
- Choose Venv
- Choose python version installed

2. Install lib:

- Check file requirements.txt to see all lib will install
- Uncomment lib torch with cpu if you want run model in cpu,
  or run code download torch with gpu if your computer have support (this way better when run optimize model)
- Run 'pip install -r requirements.txt' to install libs

3. Check your GPU (for torch with GPU)

- Run file check_cuda.py to see result, it can like this:

```
PyTorch version: 2.7.1+cu118
CUDA available: True
CUDA version: 11.8
GPU name: NVIDIA GeForce GTX 1650 with Max-Q Design
GPU memory allocated: 0.0 MB
GPU memory reserved: 0.0 MB
```
