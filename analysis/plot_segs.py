import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib.colors as mcolors

output_directory = Path('/vol/biomedic3/bglocker/mscproj/mal224/DCCL/analysis/segmentations/')
num_samples=1
torch.manual_seed(39)

name_to_fig = {'CheXmask': 'Segmentation\npre-training', 'No pretraining': 'No pre-training', 'S-MVVD-CL': 'S-MVD-CL'}

fig_unsupervised = ['No pretraining', 'SimCLR', 'DVD-CL', 'MVD-CL']
fig_supervised = ['CheXmask', 'S-DVD-CL', 'S-MVVD-CL']

colors = [(0, 0, 0, 0),
          (31/255, 50/255, 200/255, 0.6), 
          (255/255, 12/255, 14/255, 0.6)]

cmap = mcolors.ListedColormap(colors)
bounds = [0, 1, 2, 3]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

title_font_size = 22
subtitle_font_size = 20
main_title_font_size = 26

def plot_inputs(axes, images, nf_idxs, pe_idxs, std, mean):
    axes[0, 0].set_title('Input', fontsize=title_font_size)
    nf_n = len(nf_idxs)
    for i,idx in enumerate(nf_idxs):
        img = images[idx]
        img_unnorm = (img * std + mean).clip(0, 1)

        img_plot = img_unnorm.squeeze().cpu().numpy()
        axes[i, 0].text(112, 248, "No finding",
                ha='center', fontsize=subtitle_font_size)
        for j in range(len(axes[i])):
            axes[i, j].imshow(img_plot, cmap='gray')
            axes[i, j].axis('off')

    for i,idx in enumerate(pe_idxs):
        _i = i + nf_n
        img = images[idx]
        img_unnorm = (img * std + mean).clip(0, 1)

        img_plot = img_unnorm.squeeze().cpu().numpy()

        axes[_i, 0].text(112, 249, "Pleural effusion",
                ha='center', fontsize=subtitle_font_size)
        for j in range(len(axes[i])):
            axes[_i, j].imshow(img_plot, cmap='gray')
            axes[_i, j].axis('off')

def plot_label(axes, mask, nf_idxs, pe_idxs):
    axes[0, 1].set_title('Ground truth', fontsize=title_font_size)
    nf_n = len(nf_idxs)
    for i,idx in enumerate(nf_idxs):
        true_mask = mask[idx].cpu().permute(1,2,0).numpy()

        axes[i, 1].imshow(true_mask, cmap=cmap, norm=norm)

    for i,idx in enumerate(pe_idxs):
        _i = i + nf_n
        true_mask = mask[idx].cpu().permute(1,2,0).numpy()

        axes[_i, 1].imshow(true_mask, cmap=cmap, norm=norm)

def plot_column(axes, col, nf_idxs, pe_idxs, out):
    nf_n = len(nf_idxs)
    for i,idx in enumerate(nf_idxs):
        pred_mask = out[idx].argmax(dim=0).cpu().numpy()
        axes[i, col].imshow(pred_mask, cmap=cmap, norm=norm)

    for i,idx in enumerate(pe_idxs):
        _i = i + nf_n
        pred_mask = out[idx].argmax(dim=0).cpu().numpy()
        axes[_i, col].imshow(pred_mask, cmap=cmap, norm=norm)

def log(data, output_dir, model_names, idxs_nf, idxs_pe, fig_name='segmentation.png', fig_ratio = (3,3), title = 'Segmentation results'):
    mean = data['mean']
    std = data['std']

    images = data['inputs']
    mask = data['labels']
    model_outs = {name_to_fig.get(key, key): data['models'][key] for key in model_names}

    if images.ndim == 3:
        images = images.unsqueeze(1)

    in_C = images.shape[1]
    device = images.device

    mean = torch.tensor(mean).view(1, in_C, 1, 1).to(device)
    std = torch.tensor(std).view(1, in_C, 1, 1).to(device)

    N = len(idxs_nf) + len(idxs_pe)
    M = len(model_outs) + 2

    fig, axes = plt.subplots(N, M, figsize=(fig_ratio[0]*M, fig_ratio[1] * N))

    plot_inputs(axes, images, idxs_nf, idxs_pe, std, mean)
    plot_label(axes, mask, idxs_nf, idxs_pe)

    for i,(name, out) in enumerate(model_outs.items()):
        axes[0, i+2].set_title(name, fontsize=title_font_size)
        plot_column(axes, i+2, idxs_nf, idxs_pe, out)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, fig_name))
    plt.close(fig)

def main(data, num_samples = 3):
    disease = data['disease']

    mask_nf = (disease == 0)
    mask_pe = (disease == 1)

    idxs_nf_all = torch.nonzero(mask_nf, as_tuple=False).squeeze(1)
    idxs_pe_all = torch.nonzero(mask_pe, as_tuple=False).squeeze(1)

    n_nf = num_samples
    n_pe = num_samples

    idxs_nf = idxs_nf_all[torch.randperm(len(idxs_nf_all))[:n_nf]] if len(idxs_nf_all) > 0 else torch.tensor([], dtype=torch.long)
    idxs_pe = idxs_pe_all[torch.randperm(len(idxs_pe_all))[:n_pe]] if len(idxs_pe_all) > 0 else torch.tensor([], dtype=torch.long)

    log(data, output_directory, fig_unsupervised, idxs_nf, idxs_pe, fig_name='unsupervised-segmentation.png', fig_ratio=(2.8,3.1), title='Segmentation Results of Unsupervised Methods')
    log(data, output_directory, fig_supervised, idxs_nf, idxs_pe, fig_name='supervised-segmentation.png', fig_ratio=(2.8,3.3), title='Segmentation Results of Supervised Methods')

if __name__ == '__main__':
    data = torch.load(output_directory / 'out.pt')
    main(data,num_samples=num_samples)