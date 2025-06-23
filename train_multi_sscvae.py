#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for MultiSSCVAE model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os
import argparse
from tqdm import tqdm
import pandas as pd

from models import MultiSSCVAE
from utils import MultiImageNet, get_recon_loss, hoyer_metric


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_latent_loss = 0.0
    total_sparsity_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images_dict, _) in enumerate(progress_bar):
        # Move images to device
        images_dict = {k: v.to(device) for k, v in images_dict.items()}

        optimizer.zero_grad()

        # Forward pass
        recon_dict, z_dict, latent_loss_dict, dictionary = model(images_dict)

        # Compute reconstruction loss for each condition
        recon_loss = 0.0
        for cond_name in images_dict.keys():
            if cond_name != 'target':
                recon_loss += get_recon_loss(images_dict['target'], recon_dict[cond_name])

        # Compute latent loss
        latent_loss = sum(latent_loss_dict.values())

        # Compute sparsity loss
        sparsity_loss = 0.0
        for z in z_dict.values():
            sparsity_loss += hoyer_metric(z)

        # Total loss
        loss = recon_loss + 0.1 * latent_loss + 0.01 * sparsity_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_latent_loss += latent_loss.item()
        total_sparsity_loss += sparsity_loss.item()

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Recon': f'{recon_loss.item():.4f}',
            'Latent': f'{latent_loss.item():.4f}',
            'Sparsity': f'{sparsity_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_latent_loss = total_latent_loss / len(dataloader)
    avg_sparsity_loss = total_sparsity_loss / len(dataloader)

    return avg_loss, avg_recon_loss, avg_latent_loss, avg_sparsity_loss


def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0

    with torch.no_grad():
        for images_dict, _ in dataloader:
            images_dict = {k: v.to(device) for k, v in images_dict.items()}

            recon_dict, z_dict, latent_loss_dict, dictionary = model(images_dict)

            # Compute reconstruction loss
            recon_loss = 0.0
            for cond_name in images_dict.keys():
                if cond_name != 'target':
                    recon_loss += get_recon_loss(images_dict['target'], recon_dict[cond_name])

            latent_loss = sum(latent_loss_dict.values())
            loss = recon_loss + 0.1 * latent_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)

    return avg_loss, avg_recon_loss, dictionary


def main():
    parser = argparse.ArgumentParser(description='Train MultiSSCVAE')
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
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = MultiImageNet(
        root_dirs=config['data']['root_dirs'],
        mode='train',
        patch_size=config['data']['patch_size'],
        stride_size=config['data']['stride_size'],
        transform=transform
    )

    val_dataset = MultiImageNet(
        root_dirs=config['data']['root_dirs'],
        mode='val',
        patch_size=config['data']['patch_size'],
        stride_size=config['data']['stride_size'],
        transform=transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')

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
        cond=config['model']['cond'],
        device=device
    ).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    # Create save directory
    save_path = config['train']['save_path']
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'dicts'), exist_ok=True)

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(1, config['train']['epochs'] + 1):
        # Train
        train_loss, train_recon, train_latent, train_sparse = train_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # Validate
        val_loss, val_recon, dictionary = validate_epoch(model, val_loader, device)

        print(f'Epoch {epoch}:')
        print(f'  Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}')

        # Save losses
        train_losses.append({
            'epoch': epoch,
            'total_loss': train_loss,
            'recon_loss': train_recon,
            'latent_loss': train_latent,
            'sparsity_loss': train_sparse
        })

        val_losses.append({
            'epoch': epoch,
            'total_loss': val_loss,
            'recon_loss': val_recon,
        })

        # Save model
        if epoch % config['train']['save_interval'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_path, 'models', f'model_epoch_{epoch}.pth'))

            # Save dictionary
            # _, _, _, _, dictionary = model(next(iter(train_loader))[0])
            torch.save(dictionary,
                       os.path.join(save_path, 'dicts', f'dict_epoch_{epoch}.pth'))
            print("Saved the model weights and the dictionary...\n")

    # Save training history
    train_df = pd.DataFrame(train_losses)
    val_df = pd.DataFrame(val_losses)

    train_df.to_csv(os.path.join(save_path, 'training_losses.csv'), index=False)
    val_df.to_csv(os.path.join(save_path, 'validation_losses.csv'), index=False)

    print('Training completed!')


if __name__ == '__main__':
    main()
