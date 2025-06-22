# MultiSSCVAE: Multi-Condition Alignment with Sparse Coding

## Overview

MultiSSCVAE is an extension of the SSCVAE (Sparse Coding Variational Autoencoder) model designed for multi-condition alignment tasks. The main purpose is to align multiple visual conditions (such as depth maps, edge maps, etc.) to a specific target condition, enabling downstream generation models to be decoupled from condition variations.

## Key Features

- **Multi-condition Input**: Supports dictionary-format input with multiple visual conditions
- **Target Alignment**: Aligns all conditions to a specified target condition in latent space
- **Sparse Coding**: Utilizes LISTA (Learned ISTA) for sparse representation learning
- **Flexible Architecture**: Shared encoder/decoder with condition-specific processing

## Architecture

### MultiSSCVAE Model
- **Shared Encoder**: Encodes all conditions to latent space
- **LISTA Module**: Performs sparse coding on latent representations
- **Shared Decoder**: Reconstructs images from sparse codes
- **Alignment Loss**: Enforces alignment between conditions in latent space

### MultiImageNet Dataset
- **Dictionary Input**: `{"target": path1, "depth": path2, "edge": path3, ...}`
- **Flexible Conditions**: Supports arbitrary number of visual conditions
- **Standard Structure**: Compatible with ImageNet-style folder structure

## Installation

### Requirements
```bash
pip install torch torchvision
pip install scikit-image lpips pandas tqdm matplotlib pillow
```

### Project Structure
```
SSC-VAE/
├── models.py                 # Model definitions (SSCVAE, MultiSSCVAE)
├── utils.py                  # Dataset classes and utilities
├── train_multi_sscvae.py     # Training script for MultiSSCVAE
├── test_multi_sscvae.py      # Testing and evaluation script
├── demo_multi_sscvae.py      # Demo script for visualization
├── config_multi_sscvae.json  # Configuration file
└── README_MultiSSCVAE.md     # This file
```

## Usage

### 1. Data Preparation

Organize your data in the following structure:
```
dataset/
├── target_condition/
│   ├── train/
│   │   ├── class1/
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   └── class2/
│   ├── val/
│   └── test/
├── depth_maps/
│   ├── train/
│   ├── val/
│   └── test/
└── edge_maps/
    ├── train/
    ├── val/
    └── test/
```

### 2. Configuration

Edit `config_multi_sscvae.json`:
```json
{
  "data": {
    "root_dirs": {
      "target": "/path/to/target_condition",
      "depth": "/path/to/depth_maps",
      "edge": "/path/to/edge_maps"
    },
    "patch_size": 256,
    "stride_size": 256,
    "batch_size": 16
  },
  "model": {
    "in_channels": 3,
    "hid_channels_1": 64,
    "hid_channels_2": 128,
    "out_channels": 256,
    "down_samples": 3,
    "num_atoms": 512,
    "num_dims": 256,
    "num_iters": 10
  },
  "loss": {
    "recon_weight": 1.0,
    "latent_weight": 0.1,
    "alignment_weight": 1.0,
    "sparsity_weight": 0.01
  }
}
```

### 3. Training

```bash
python train_multi_sscvae.py --config config_multi_sscvae.json
```

### 4. Testing

```bash
python test_multi_sscvae.py --config config_multi_sscvae.json
```

### 5. Demo

```bash
# Run alignment demo
python demo_multi_sscvae.py --config config_multi_sscvae.json --mode demo --num_samples 5

# Analyze model capacity
python demo_multi_sscvae.py --config config_multi_sscvae.json --mode analyze
```

## Model Components

### MultiSSCVAE Class

```python
from models import MultiSSCVAE

# Initialize model
model = MultiSSCVAE(
    in_channels=3,
    hid_channels_1=64,
    hid_channels_2=128,
    out_channels=256,
    down_samples=3,
    num_groups=4,
    num_atoms=512,
    num_dims=256,
    num_iters=10,
    device='cuda'
)

# Forward pass
images_dict = {
    'target': target_images,    # [B, C, H, W]
    'depth': depth_images,      # [B, C, H, W]
    'edge': edge_images         # [B, C, H, W]
}

recon_dict, z_dict, latent_loss, alignment_loss, sparsity_loss = model(images_dict)

# Align conditions to target
aligned_dict = model.align_to_target(images_dict)
```

### MultiImageNet Dataset

```python
from utils import MultiImageNet
from torchvision import transforms

# Create dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = MultiImageNet(
    root_dirs={
        'target': '/path/to/target',
        'depth': '/path/to/depth',
        'edge': '/path/to/edge'
    },
    mode='train',
    patch_size=256,
    stride_size=256,
    transform=transform
)

# Get sample
images_dict, path = dataset[0]
# images_dict: {'target': tensor, 'depth': tensor, 'edge': tensor}
```

## Loss Functions

The model optimizes multiple loss components:

1. **Reconstruction Loss**: L2 loss between input and reconstructed images
2. **Latent Loss**: Regularization on latent representations
3. **Alignment Loss**: L2 loss between non-target and target latent codes
4. **Sparsity Loss**: L1 regularization on sparse codes

```python
total_loss = (recon_weight * recon_loss + 
              latent_weight * latent_loss + 
              alignment_weight * alignment_loss + 
              sparsity_weight * sparsity_loss)
```

## Evaluation Metrics

The testing script computes:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **NMI**: Normalized Mutual Information
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **Alignment Loss**: Latent space alignment quality
- **Sparsity Statistics**: Activation patterns and sparsity levels

## Applications

### 1. Condition Translation
Translate between different visual conditions while preserving semantic content.

### 2. Domain Adaptation
Align different visual domains to a common representation space.

### 3. Multi-modal Generation
Enable generation models to work with various input conditions by aligning them to a learned target space.

### 4. Data Augmentation
Generate aligned condition pairs for training downstream models.

## Tips and Best Practices

### 1. Data Preparation
- Ensure all conditions have corresponding images with same filenames
- Use consistent image sizes and formats
- Verify data quality and alignment

### 2. Training
- Start with lower alignment weight and gradually increase
- Monitor sparsity levels (target: 0.1-0.3)
- Use learning rate scheduling for better convergence

### 3. Hyperparameter Tuning
- `alignment_weight`: Controls alignment strength (0.5-2.0)
- `sparsity_weight`: Controls sparsity level (0.001-0.1)
- `num_iters`: LISTA iterations (5-20)
- `num_atoms`: Dictionary size (256-1024)

### 4. Evaluation
- Use multiple metrics for comprehensive evaluation
- Visualize alignment results regularly
- Check sparsity statistics for proper sparse coding

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size
   - Use gradient checkpointing
   - Reduce image resolution

2. **Poor Alignment**
   - Increase alignment weight
   - Check data correspondence
   - Verify target condition quality

3. **Over-sparsity**
   - Reduce sparsity weight
   - Increase number of atoms
   - Check LISTA iterations

4. **Training Instability**
   - Use gradient clipping
   - Reduce learning rate
   - Add batch normalization

## Citation

If you use this code in your research, please cite:

```bibtex
@article{multisscvae2024,
  title={MultiSSCVAE: Multi-Condition Alignment with Sparse Coding},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.