import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import argparse
import json
from types import SimpleNamespace
import csv

from utils import GrayDataset, UltrasoundDataset, MiniImagenet, hoyer_metric, compute_indicators, slice_image, recon_image
from models import SSCVAE
from visualization import plot_images, plot_dict_tsne


'''hyperparameters'''
parser = argparse.ArgumentParser(description='program args')
parser.add_argument('-c', '--config', type=str, required=True, help='config file path')
args = parser.parse_args()
with open(args.config, 'r') as json_data:
    config_data = json.load(json_data)
data_args = SimpleNamespace(**config_data['data'])
model_args = SimpleNamespace(**config_data['model'])
train_args = SimpleNamespace(**config_data['train'])
test_args = SimpleNamespace(**config_data['test'])


'''make dir'''
model_fold_path = os.path.join(train_args.save_path, 'models')
image_fold_path = os.path.join(train_args.save_path, 'images')


'''dataset'''
data_transform = {
    "test": transforms.Compose([transforms.ToTensor()])
}

if data_args.dataset == 'gray':
    test_dataset = GrayDataset(root_dir=data_args.root_dir,
                               seed=data_args.seed,
                               train_ratio=data_args.train_ratio,
                               val_ratio=data_args.val_ratio,
                               mode="test",
                               patch_size=data_args.patch_size,
                               stride_size=data_args.stride_size,
                               transform=data_transform["test"])
elif data_args.dataset == 'ultrasound':
    test_dataset = UltrasoundDataset(root_dir=data_args.root_dir,
                                     quality=data_args.quality,
                                     seed=data_args.seed,
                                     train_ratio=data_args.train_ratio,
                                     val_ratio=data_args.val_ratio,
                                     mode="test",
                                     transform=data_transform["test"])
elif data_args.dataset == 'imagenet':
    test_dataset = MiniImagenet(root_dir=data_args.root_dir,
                                mode='test',
                                patch_size=data_args.patch_size,
                                stride_size=data_args.stride_size,
                                transform=data_transform["test"])
else:
    raise ValueError("dataset: {} isn't allowed.".format(data_args.dataset))

test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=False,
                         pin_memory=True)

test_image_num = len(test_dataset)
print("test_image_num:", test_image_num)


'''model'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SSCVAE(in_channels=model_args.in_channels,
               hid_channels_1=model_args.hid_channels_1,
               hid_channels_2=model_args.hid_channels_2,
               out_channels=model_args.out_channels,
               down_samples=model_args.down_samples,
               num_groups=model_args.num_groups,
               num_atoms=model_args.num_atoms,
               num_dims=model_args.num_dims,
               num_iters=model_args.num_iters,
               device=device).to(device)

load_path = os.path.join(model_fold_path, f'model{test_args.model_id:d}.pt')
model.load_state_dict(torch.load(load_path, map_location=device))
model.eval()


'''test'''
csv_filename = os.path.join(train_args.save_path, 'testing_indicators.csv')
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Name', 'Sparsity', 'PSNR', 'SSIM', 'NMI', 'LPIPS']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

with torch.no_grad():
    for index, (image_origin, name) in enumerate(test_loader):
        name = name[0]
        image_origin = image_origin.to(device)

        if data_args.dataset == 'gray' or data_args.dataset == 'imagenet':
            ori_shape = image_origin.shape
            patches = slice_image(image_origin, data_args.patch_size, data_args.stride_size).to(device)

            bs, _, _, _ = patches.shape
            if bs > 16:
                continue

            patches_recon, z, _, dictionary = model(patches)
            image_recon = recon_image(patches_recon, ori_shape, data_args.patch_size, data_args.stride_size).to(device)
        else:
            image_recon, z, _, dictionary = model(image_origin)

            plot_dict_tsne(dictionary, './', 'sscvae_tsne.png')
        break

        sparsity = hoyer_metric(z).item()
        PSNR, SSIM, NMI, LPIPS = compute_indicators(image_origin, image_recon)

        # if (index + 1) % 100 == 0:
        plot_images(image_origin,
                    image_recon,
                    image_fold_path,
                    name,
                    channels=model_args.in_channels)

        # write result
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Name': name,
                             'Sparsity': sparsity,
                             'PSNR': PSNR,
                             'SSIM': SSIM,
                             'NMI': NMI,
                             'LPIPS': LPIPS})
