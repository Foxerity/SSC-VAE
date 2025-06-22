#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for MultiSSCVAE - Multi-condition Alignment
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from models import MultiSSCVAE
from utils import MultiImageNet

def visualize_alignment(images_dict, aligned_dict, save_path, sample_idx=0):
    """
    Visualize original conditions and their alignment to target
    """
    condition_names = list(images_dict.keys())
    num_conditions = len(condition_names)
    
    fig, axes = plt.subplots(2, num_conditions, figsize=(4*num_conditions, 8))
    
    def tensor_to_numpy(tensor):
        tensor = tensor[sample_idx].cpu().detach()
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.clamp(0, 1).numpy()
        if tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
            return tensor, 'gray'
        else:
            return tensor, None
    
    # Plot original images
    for i, cond_name in enumerate(condition_names):
        img_np, cmap = tensor_to_numpy(images_dict[cond_name])
        axes[0, i].imshow(img_np, cmap=cmap)
        axes[0, i].set_title(f'Original {cond_name}')
        axes[0, i].axis('off')
    
    # Plot aligned images
    for i, cond_name in enumerate(condition_names):
        img_np, cmap = tensor_to_numpy(aligned_dict[cond_name])
        axes[1, i].imshow(img_np, cmap=cmap)
        axes[1, i].set_title(f'Aligned {cond_name}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'alignment_demo_sample_{sample_idx}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def demo_alignment(config_path, model_path, num_samples=5):
    """
    Demonstrate multi-condition alignment
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Create test dataset
    test_dataset = MultiImageNet(
        root_dirs=config['data']['root_dirs'],
        mode='test',
        patch_size=config['data']['patch_size'],
        stride_size=config['data']['stride_size'],
        transform=transform
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one sample at a time for demo
        shuffle=True,
        num_workers=1
    )
    
    print(f'Test dataset size: {len(test_dataset)}')
    
    # Create model
    model = MultiSSCVAE(
        in_channels=config['model']['in_channels'],
        hid_channels_1=config['model']['hid_channels_1'],
        hid_channels_2=config['model']['hid_channels_2'],
        out_channels=config['model']['out_channels'],
        down_samples=config['model']['down_samples'],
        num_groups=config['model']['num_groups'],
        num_atoms=config['model']['num_atoms'],
        num_dims=config['model']['num_dims'],
        num_iters=config['model']['num_iters'],
        device=device
    ).to(device)
    
    # Load trained model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Loaded model from {model_path}')
    else:
        print(f'Model file {model_path} not found! Using random weights for demo.')
    
    # Create demo output directory
    demo_path = './demo_results'
    os.makedirs(demo_path, exist_ok=True)
    
    model.eval()
    
    print(f'\nGenerating {num_samples} alignment demos...')
    
    with torch.no_grad():
        for sample_idx, (images_dict, paths) in enumerate(test_loader):
            if sample_idx >= num_samples:
                break
            
            print(f'Processing sample {sample_idx + 1}/{num_samples}')
            
            # Move images to device
            images_dict = {k: v.to(device) for k, v in images_dict.items()}
            
            # Get model outputs
            recon_dict, z_dict, latent_loss, alignment_loss, sparsity_loss = model(images_dict)
            
            # Get aligned results
            aligned_dict = model.align_to_target(images_dict)
            
            # Print losses
            print(f'  Latent Loss: {latent_loss.item():.4f}')
            print(f'  Alignment Loss: {alignment_loss.item():.4f}')
            print(f'  Sparsity Loss: {sparsity_loss.item():.4f}')
            
            # Visualize alignment
            visualize_alignment(images_dict, aligned_dict, demo_path, sample_idx=0)
            
            # Print sparsity statistics
            print(f'  Sparsity statistics:')
            for cond_name, z in z_dict.items():
                sparsity = (torch.abs(z) < 0.01).float().mean().item()
                print(f'    {cond_name}: {sparsity:.3f}')
            
            print(f'  Sample path: {paths[0]}')
            print()
    
    print(f'Demo completed! Results saved in {demo_path}')

def analyze_model_capacity(config_path):
    """
    Analyze model capacity and parameter count
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cpu')  # Use CPU for analysis
    
    # Create model
    model = MultiSSCVAE(
        in_channels=config['model']['in_channels'],
        hid_channels_1=config['model']['hid_channels_1'],
        hid_channels_2=config['model']['hid_channels_2'],
        out_channels=config['model']['out_channels'],
        down_samples=config['model']['down_samples'],
        num_groups=config['model']['num_groups'],
        num_atoms=config['model']['num_atoms'],
        num_dims=config['model']['num_dims'],
        num_iters=config['model']['num_iters'],
        device=device
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('\nModel Analysis:')
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)')
    
    # Analyze each component
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    lista_params = sum(p.numel() for p in model.lista.parameters())
    
    print(f'\nComponent breakdown:')
    print(f'Encoder: {encoder_params:,} parameters ({encoder_params/total_params*100:.1f}%)')
    print(f'Decoder: {decoder_params:,} parameters ({decoder_params/total_params*100:.1f}%)')
    print(f'LISTA: {lista_params:,} parameters ({lista_params/total_params*100:.1f}%)')
    
    # Test forward pass
    print('\nTesting forward pass...')
    
    # Create dummy input
    dummy_input = {
        'target': torch.randn(1, 3, 256, 256),
        'depth': torch.randn(1, 3, 256, 256),
        'edge': torch.randn(1, 3, 256, 256)
    }
    
    try:
        with torch.no_grad():
            recon_dict, z_dict, latent_loss, alignment_loss, sparsity_loss = model(dummy_input)
        
        print('Forward pass successful!')
        print(f'Output shapes:')
        for cond_name, recon in recon_dict.items():
            print(f'  {cond_name}: {recon.shape}')
        
        print(f'Sparse code shapes:')
        for cond_name, z in z_dict.items():
            print(f'  {cond_name}: {z.shape}')
            
    except Exception as e:
        print(f'Forward pass failed: {e}')

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MultiSSCVAE Demo')
    parser.add_argument('--config', type=str, default='config_multi_sscvae.json', 
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='', 
                       help='Path to trained model (optional)')
    parser.add_argument('--mode', type=str, choices=['demo', 'analyze'], default='demo',
                       help='Demo mode: demo or analyze')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples for demo')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_alignment(args.config, args.model, args.num_samples)
    elif args.mode == 'analyze':
        analyze_model_capacity(args.config)

if __name__ == '__main__':
    main()