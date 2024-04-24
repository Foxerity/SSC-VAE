import torch
from torch.utils.data import Dataset
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_mutual_information
import lpips

import os
from PIL import Image
import pandas as pd
import random


class GrayDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 seed: int,
                 train_ratio: float,
                 val_ratio: float,
                 mode: str,
                 patch_size: int,
                 stride_size: int,
                 transform=None):
        names = os.listdir(root_dir)
        random.seed(seed)
        random.shuffle(names)
        
        n = len(names)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)

        if mode == 'train':
            self.img_names = names[: train_n]
        elif mode == 'val':
            self.img_names = names[train_n : train_n + val_n]
        elif mode == 'test':
            self.img_names = names[train_n + val_n :]
        else:
            raise ValueError("mode: {} isn't allowed.".format(mode))

        self.root_dir = root_dir
        self.mode = mode
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.transform = transform

        self.indices = []
        for name in self.img_names:
            img_path = os.path.join(root_dir, name)
            img = Image.open(img_path)
            width, height = img.size
            if width >= patch_size and height >= patch_size:
                if mode == 'test':
                    self.indices.append(name)
                else:
                    num_patches_x = (width - patch_size) // stride_size + 1
                    num_patches_y = (height - patch_size) // stride_size + 1
                    for i in range(num_patches_x):
                        for j in range(num_patches_y):
                            x = i * self.stride_size
                            y = j * self.stride_size
                            self.indices.append((name, x, y))

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        if self.mode == 'test':
            name = self.indices[index]
            img_path = os.path.join(self.root_dir, name)
            img = Image.open(img_path)

            width, height = img.size
            num_patches_x = (width - self.patch_size) // self.stride_size + 1
            num_patches_y = (height - self.patch_size) // self.stride_size + 1
            cropped_width = (num_patches_x - 1) * self.stride_size + self.patch_size
            cropped_height = (num_patches_y - 1) * self.stride_size + self.patch_size
            
            image = img.crop((0, 0, cropped_width, cropped_height))
        else:
            name, x, y = self.indices[index]
            img_path = os.path.join(self.root_dir, name)
            img = Image.open(img_path)

            image = img.crop((x, y, x + self.patch_size, y + self.patch_size))
            name = name.replace(".jpg", f"_{x}_{y}.jpg")

        if image.mode != 'L':
            raise ValueError("image: {} isn't L mode.".format(name))
        if self.transform is not None:
            image = self.transform(image)

        return image, name

class UltrasoundDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 quality: str,
                 seed: int,
                 train_ratio: float,
                 val_ratio: float,
                 mode: str,
                 transform=None):
        organs = ['breast', 'carotid', 'kidney', 'liver', 'thyroid']
        all_image_infos = []
        for organ in organs:
            image_path = os.path.join(root_dir, organ, quality)
            image_names = os.listdir(image_path)
            image_infos = []
            for image_name in image_names:
                image_info = {
                    'organ': organ,
                    'name': image_name
                }
                image_infos.append(image_info)
            all_image_infos.extend(image_infos)
        random.seed(seed)
        random.shuffle(all_image_infos)

        n = len(all_image_infos)
        train_n = int(n * train_ratio) 
        val_n = int(n * val_ratio)

        if mode == 'train':
            self.image_infos = all_image_infos[: train_n]
        elif mode == 'val':
            self.image_infos = all_image_infos[train_n : train_n + val_n]
        elif mode == 'test':
            self.image_infos = all_image_infos[train_n + val_n :]
        else:
            raise ValueError("mode: {} isn't allowed.".format(mode))

        self.root_dir = root_dir
        self.quality = quality
        self.transform = transform

    def __len__(self):
        return len(self.image_infos)
    
    def __getitem__(self, index):
        organ = self.image_infos[index]['organ']
        name_ = self.image_infos[index]['name']
        name = organ + '_' + name_
        img_path = os.path.join(self.root_dir, organ, self.quality, name_)
        img = Image.open(img_path)

        if img.mode != 'L':
            raise ValueError("image: {} isn't L mode.".format(name))
        if self.transform is not None:
            img = self.transform(img)

        return img, name

class MiniImagenet(Dataset):
    def __init__(self,
                 root_dir: str,
                 mode: str,
                 patch_size: int,
                 stride_size: int,
                 transform=None):
        csv_name = mode + '.csv'
        images_dir = os.path.join(root_dir, "images")
        csv_path = os.path.join(root_dir, csv_name)
        csv_data = pd.read_csv(csv_path)

        self.images_dir = images_dir
        self.mode = mode
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.transform = transform

        names = [i for i in csv_data["filename"].values]

        self.indices = []
        for name in names:
            img_path = os.path.join(images_dir, name)
            img = Image.open(img_path)
            width, height = img.size
            if width >= patch_size and height >= patch_size:
                self.indices.append(name)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        name = self.indices[index]
        img_path = os.path.join(self.images_dir, name)
        img = Image.open(img_path)

        if self.mode == 'test':
            width, height = img.size
            num_patches_x = (width - self.patch_size) // self.stride_size + 1
            num_patches_y = (height - self.patch_size) // self.stride_size + 1
            cropped_width = (num_patches_x - 1) * self.stride_size + self.patch_size
            cropped_height = (num_patches_y - 1) * self.stride_size + self.patch_size
            
            image = img.crop((0, 0, cropped_width, cropped_height))
        else:
            image = img

        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(name))
        if self.transform is not None:
            image = self.transform(image)

        return image, name

def get_noise(image_origin, sigma):
    image_noise = torch.empty_like(image_origin)
    image_noise.copy_(image_origin)
    noise = torch.randn_like(image_origin)
    image_noise = image_noise + sigma * noise / 255
    image_noise = torch.clamp(image_noise, 0, 1)
    return image_noise

def slice_image(image, patch_size, stride):
    n, c, h, w = image.shape 
    patch_h = patch_size 
    patch_w = patch_size

    n_patches_h = (h-patch_h) // stride + 1
    n_patches_w = (w-patch_w) // stride + 1 

    out_shape = (n_patches_h * n_patches_w, c, patch_h, patch_w)  
    patches = torch.zeros(out_shape)
    
    index = 0
    for hi in range(0, h-patch_h+1, stride):
        for wi in range(0, w-patch_w+1, stride):  
            patches[index] = image[:, :, hi:hi+patch_h, wi:wi+patch_w]
            index += 1

    return patches

def recon_image(patches, ori_shape, patch_size, stride):
    n_patches, c, h, w = patches.shape  
    n, c, orig_h, orig_w = ori_shape

    reconstructed = torch.zeros(ori_shape)

    patch_h = patch_size
    patch_w = patch_size

    idx = 0
    for hi in range(0, orig_h-patch_h+1, stride):
        for wi in range(0, orig_w-patch_w+1, stride):
            reconstructed[:,:,hi:hi+patch_h,wi:wi+patch_w] = patches[idx]
            idx += 1

    return reconstructed

def get_recon_loss(images_origin, images_recon):
    recon_loss = (images_origin - images_recon).pow(2).mean()
    return recon_loss

def hoyer_metric(z):
    b, K, h, w = z.shape
    K = torch.tensor(K)
    
    l1_norm = torch.norm(z, p=1, dim=1, keepdim=True)  # [B, 1, H, W]
    l2_norm = torch.norm(z, p=2, dim=1, keepdim=True)  # [B, 1, H, W]

    sparsity_score = (torch.sqrt(K) - l1_norm / l2_norm) / (torch.sqrt(K) - 1)
    hoyer_metric_value = torch.mean(sparsity_score)

    return hoyer_metric_value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='alex').to(device)

def compute_indicators(image_true, image_restored):
    # [1, 1, H, W]
    LPIPS = lpips_model(image_true[0], image_restored[0])

    image_true = image_true[0][0].detach().cpu().numpy()
    image_restored = image_restored[0][0].detach().cpu().numpy()

    PSNR = peak_signal_noise_ratio(image_true, image_restored, data_range=1)
    SSIM = structural_similarity(image_true, image_restored, data_range=1)
    NMI = normalized_mutual_information(image_true, image_restored)

    return PSNR, SSIM, NMI, LPIPS.item()
