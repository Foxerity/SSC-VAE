import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_image(image, image_fold_path, image_name, channels):
    if channels == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    os.makedirs(os.path.dirname(os.path.join(image_fold_path, image_name)), exist_ok=True)
    plt.savefig(os.path.join(image_fold_path, image_name), dpi=300)
    plt.close()
    
def plot_images(image_origin, image_recon, image_fold_path, image_name, channels):
    image_origin = image_origin[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
    plot_image(image_origin, os.path.join(image_fold_path, 'origin'), image_name, channels)

    image_recon = image_recon[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
    plot_image(image_recon, os.path.join(image_fold_path, 'recon'), image_name, channels)

def plot_dict(dictionary, image_fold_path, image_name):
    atom_size, num_atoms = dictionary.shape
    dictionary = dictionary.detach().cpu().numpy()

    fig, ax = plt.subplots(16, 32, figsize=(16, 8))

    for i in range(num_atoms):
        row = i // 32
        col = i % 32
        atom = dictionary[:, i].reshape((16, 16))
        ax[row, col].imshow(atom, cmap='gray')
        ax[row, col].axis('off')

    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    os.makedirs(os.path.dirname(os.path.join(image_fold_path, image_name)), exist_ok=True)
    plt.savefig(os.path.join(image_fold_path, image_name))
    plt.close()

def plot_dict_tsne(dictionary, image_fold_path, image_name):
    tsne = TSNE(n_components=2, random_state=42)
    dictionary_2d = tsne.fit_transform(dictionary.T.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.scatter(dictionary_2d[:, 0], dictionary_2d[:, 1], s=10)
    plt.title('Visualization of Dictionary')

    os.makedirs(os.path.dirname(os.path.join(image_fold_path, image_name)), exist_ok=True)
    plt.savefig(os.path.join(image_fold_path, image_name))
    plt.close()
