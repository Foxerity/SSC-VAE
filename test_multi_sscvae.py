#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing script for MultiSSCVAE model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image

from models import MultiSSCVAE
from utils import MultiImageNet, compute_indicators

def test_model(model, dataloader, device, save_path):
    model.eval()
    
    all_indicators = []
    condition_names = None
    
    os.makedirs(os.path.join(save_path, 'test_images'), exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images_dict, paths) in enumerate(tqdm(dataloader, desc='Testing')):
            if condition_names is None:
                condition_names = list(images_dict.keys())
            
            # Move images to device
            images_dict = {k: v.to(device) for k, v in images_dict.items()}
            
            # Get aligned results
            aligned_dict = model.align_to_target(images_dict)
            
            # Compute indicators for each condition alignment to target
            for i in range(images_dict['target'].size(0)):
                target_img = images_dict['target'][i:i+1]
                
                for cond_name in condition_names:
                    if cond_name != 'target':
                        aligned_img = aligned_dict[cond_name][i:i+1]
                        
                        # Compute quality indicators
                        psnr, ssim, nmi, lpips = compute_indicators(target_img, aligned_img)
                        
                        all_indicators.append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'condition': cond_name,
                            'PSNR': psnr,
                            'SSIM': ssim,
                            'NMI': nmi,
                            'LPIPS': lpips,
                            'path': paths[i]
                        })
                
                # Save sample images for visualization
                if batch_idx < 5:  # Save first 5 batches
                    save_sample_images(images_dict, aligned_dict, i, batch_idx, save_path)
    
    return all_indicators

def save_sample_images(images_dict, aligned_dict, sample_idx, batch_idx, save_path):
    """Save sample images for visualization"""
    
    def tensor_to_pil(tensor):
        # Convert tensor to PIL Image
        tensor = tensor.squeeze(0).cpu()
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = (tensor * 255).clamp(0, 255).byte().numpy()
        if tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
            return Image.fromarray(tensor, mode='L')
        else:
            return Image.fromarray(tensor, mode='RGB')
    
    sample_dir = os.path.join(save_path, 'test_images', f'batch_{batch_idx}_sample_{sample_idx}')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Save original images
    for cond_name, img_tensor in images_dict.items():
        img = tensor_to_pil(img_tensor[sample_idx:sample_idx+1])
        img.save(os.path.join(sample_dir, f'original_{cond_name}.png'))
    
    # Save aligned images
    for cond_name, img_tensor in aligned_dict.items():
        img = tensor_to_pil(img_tensor[sample_idx:sample_idx+1])
        img.save(os.path.join(sample_dir, f'aligned_{cond_name}.png'))

def compute_alignment_metrics(model, dataloader, device):
    """Compute alignment quality metrics"""
    model.eval()
    
    total_alignment_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images_dict, _ in tqdm(dataloader, desc='Computing alignment metrics'):
            images_dict = {k: v.to(device) for k, v in images_dict.items()}
            
            # Forward pass to get alignment loss
            _, _, _, alignment_loss, _ = model(images_dict)
            
            total_alignment_loss += alignment_loss.item() * images_dict['target'].size(0)
            total_samples += images_dict['target'].size(0)
    
    avg_alignment_loss = total_alignment_loss / total_samples
    return avg_alignment_loss

def analyze_sparse_codes(model, dataloader, device, save_path):
    """Analyze sparse code statistics"""
    model.eval()
    
    all_sparsity_scores = {}
    all_activation_patterns = {}
    
    with torch.no_grad():
        for images_dict, _ in tqdm(dataloader, desc='Analyzing sparse codes'):
            images_dict = {k: v.to(device) for k, v in images_dict.items()}
            
            _, z_dict, _, _, _ = model(images_dict)
            
            for cond_name, z in z_dict.items():
                # Compute sparsity (percentage of near-zero activations)
                sparsity = (torch.abs(z) < 0.01).float().mean().item()
                
                if cond_name not in all_sparsity_scores:
                    all_sparsity_scores[cond_name] = []
                all_sparsity_scores[cond_name].append(sparsity)
                
                # Compute activation patterns
                activation_pattern = (torch.abs(z) > 0.01).float().mean(dim=(0, 2, 3)).cpu().numpy()
                
                if cond_name not in all_activation_patterns:
                    all_activation_patterns[cond_name] = []
                all_activation_patterns[cond_name].append(activation_pattern)
    
    # Save sparsity analysis
    sparsity_results = {}
    for cond_name, scores in all_sparsity_scores.items():
        sparsity_results[cond_name] = {
            'mean_sparsity': np.mean(scores),
            'std_sparsity': np.std(scores),
            'min_sparsity': np.min(scores),
            'max_sparsity': np.max(scores)
        }
    
    # Save activation patterns
    activation_results = {}
    for cond_name, patterns in all_activation_patterns.items():
        patterns = np.array(patterns)
        activation_results[cond_name] = {
            'mean_activation': patterns.mean(axis=0).tolist(),
            'std_activation': patterns.std(axis=0).tolist()
        }
    
    # Save results
    with open(os.path.join(save_path, 'sparsity_analysis.json'), 'w') as f:
        json.dump({
            'sparsity_scores': sparsity_results,
            'activation_patterns': activation_results
        }, f, indent=2)
    
    return sparsity_results, activation_results

def main():
    parser = argparse.ArgumentParser(description='Test MultiSSCVAE')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
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
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=4
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
    save_path = config['train']['save_path']
    model_id = config['test']['model_id']
    model_path = os.path.join(save_path, 'models', f'model_epoch_{model_id}.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Loaded model from {model_path}')
    else:
        print(f'Model file {model_path} not found!')
        return
    
    # Test model
    print('Testing model...')
    indicators = test_model(model, test_loader, device, save_path)
    
    # Compute alignment metrics
    print('Computing alignment metrics...')
    avg_alignment_loss = compute_alignment_metrics(model, test_loader, device)
    
    # Analyze sparse codes
    print('Analyzing sparse codes...')
    sparsity_results, activation_results = analyze_sparse_codes(model, test_loader, device, save_path)
    
    # Save test results
    indicators_df = pd.DataFrame(indicators)
    indicators_df.to_csv(os.path.join(save_path, 'testing_indicators.csv'), index=False)
    
    # Compute and save summary statistics
    summary_stats = {}
    for cond_name in indicators_df['condition'].unique():
        cond_data = indicators_df[indicators_df['condition'] == cond_name]
        summary_stats[cond_name] = {
            'PSNR_mean': cond_data['PSNR'].mean(),
            'PSNR_std': cond_data['PSNR'].std(),
            'SSIM_mean': cond_data['SSIM'].mean(),
            'SSIM_std': cond_data['SSIM'].std(),
            'NMI_mean': cond_data['NMI'].mean(),
            'NMI_std': cond_data['NMI'].std(),
            'LPIPS_mean': cond_data['LPIPS'].mean(),
            'LPIPS_std': cond_data['LPIPS'].std()
        }
    
    summary_stats['alignment_loss'] = avg_alignment_loss
    
    with open(os.path.join(save_path, 'test_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Print summary
    print('\nTest Results Summary:')
    print(f'Average Alignment Loss: {avg_alignment_loss:.4f}')
    print('\nCondition Alignment Quality:')
    for cond_name, stats in summary_stats.items():
        if cond_name != 'alignment_loss':
            print(f'  {cond_name}:')
            print(f'    PSNR: {stats["PSNR_mean"]:.2f} ± {stats["PSNR_std"]:.2f}')
            print(f'    SSIM: {stats["SSIM_mean"]:.3f} ± {stats["SSIM_std"]:.3f}')
            print(f'    LPIPS: {stats["LPIPS_mean"]:.3f} ± {stats["LPIPS_std"]:.3f}')
    
    print('\nSparsity Analysis:')
    for cond_name, stats in sparsity_results.items():
        print(f'  {cond_name}: {stats["mean_sparsity"]:.3f} ± {stats["std_sparsity"]:.3f}')
    
    print('Testing completed!')

if __name__ == '__main__':
    main()