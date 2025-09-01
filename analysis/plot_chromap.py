import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os
from PIL import Image
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

legend_size = 19
subtitle_size = 21

output = Path('/vol/biomedic3/bglocker/mscproj/mal224/DCCL/analysis/chromap')

runs = [
    ('DVD-CL', '/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/unsupervised_cf_dense_two_noflip_random/2025-08-14_17-03-42/pretraining/visualizations/step-153940/one_image/blended'),
    ('S-DVD-CL', '/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/supervised_cf_dense_two_noflip_random/2025-08-21_12-38-44/pretraining/visualizations/step-115455/one_image/blended'),
    ('MVD-CL', '/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/unsupervised_cf_dense_all_noflip_random/2025-08-15_23-28-38/pretraining/visualizations/step-115455/one_image/blended'),
    ('S-MVD-CL','/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full/supervised_cf_dense_all_noflip_random/2025-08-14_16-59-09/pretraining/visualizations/step-115455/one_image/blended')
]

images = ['view_og.png','view_pe.png', 'view_sc.png', 'view_sc_pe.png']
names = ['View 1 (Anchor)', 'View 2', 'View 3', 'View 4']

def get_umap(ax, folder):
    image_path = Path(folder).parent.parent / 'all' / 'embeddings.png'
    img = Image.open(image_path)

    crop_box = (60, 40, img.width - 165, img.height - 50)
    cropped_img = img.crop(crop_box)
    cropped_img = cropped_img.resize((500,500))
    img_array = np.array(cropped_img)

    im = ax.imshow(img_array)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Background',
            markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Left lung',
            markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Right lung',
            markerfacecolor='yellow', markersize=10)
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        frameon=True,
        fontsize=legend_size,
        labelspacing=0.2,
        handlelength=1,
        handletextpad=0.3,
        columnspacing=0.5
    )
    
    ax.set_title('UMAP', fontsize=subtitle_size)
    ax.axis("off")

def plot_run(folder, method):
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 3, figure=fig, wspace=0.05, hspace=0.05, width_ratios=[2,1,1], height_ratios=[1,1], left=0.02, right=0.98,top=0.95,bottom=0.02)

    ax_big = fig.add_subplot(gs[:, 0])
    get_umap(ax_big, folder)

    pos = ax_big.get_position()
    ax_big.set_position([pos.x0, pos.y0 + 0.04, pos.width, pos.height])

    for i, (img, name) in enumerate(zip(images, names)):
        row = i // 2
        col = i % 2 + 1
        ax = fig.add_subplot(gs[row, col])
        img_path = os.path.join(folder, img)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(name, fontsize=subtitle_size)
        ax.axis("off")
    
    plt.savefig(output / f'{method.lower()[:-3].replace('-','')}-chromap.png')

def main():
    for name, folder in runs:
        plot_run(folder, name)

if __name__ == '__main__':
    main()