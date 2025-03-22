# UNet with Self-Attention 

A PyTorch Lightning implementation (as a tutorial) of UNet with attention mechanism for semantic segmentation on the KITTI dataset.

## Features

- UNet architecture with attention mechanisms
- PyTorch Lightning integration for clean and scalable training, including Distributed Data Parallel (DDP) support for multi-GPU training
- TensorBoard integration for:
  - Real-time metric tracking (loss, dice score, mIoU, and inferenced semantic maps)
  - Training progress visualization
  - Performance monitoring
- Hydra configuration system
- Docker and SLURM support for cluster deployment
