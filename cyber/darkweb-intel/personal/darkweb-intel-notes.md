## Why WSL2 and not Windows

Almost everything in ML was designed for Linux. PyTorch, CUDA tools, Docker, most training scripts — all of it assumes Linux. WSL2 runs a real Linux kernel on your Windows machine with direct access to your GPU through NVIDIA's driver. You get Linux tooling without buying a new computer.

The driver lives on the Windows side. You don't install a GPU driver inside WSL2 — that's a common mistake that breaks things. The CUDA toolkit and your ML frameworks go on the WSL2 side. They reach the GPU through NVIDIA's bridge between the two.