import torch
import matplotlib.pyplot as plt
from transfer.evaluation.base_evaluation import BaseEvaluator
import os

class VisualizationEvaluator(BaseEvaluator):
    def __init__(
            self, 
            eval_name,
            eval_freq, 
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            directory = 'visualizations',
            seperate_disease=False,
            num_samples=4,
            dataset=None):
        super().__init__(eval_name, eval_freq)
        self.mean = mean
        self.std = std
        self.directory = directory
        self.seperate_disease = seperate_disease
        self.num_samples = num_samples
    
    def evaluate(self, model, out, mask):
        return out, mask
    
    def _log(self, step, input, results, logger):
        if not os.path.exists(os.path.join(self.output_dir, self.directory)):
            os.mkdir(os.path.join(self.output_dir, self.directory))
            if self.seperate_disease:
                os.mkdir(os.path.join(self.output_dir, self.directory, 'nf'))
                os.mkdir(os.path.join(self.output_dir, self.directory, 'pe'))
        
        images = input['x']
        disease = input['disease']
        if self.seperate_disease:
            mask_nf = (disease == 0)
            mask_pe = (disease == 1)

            idxs_nf_all = torch.nonzero(mask_nf, as_tuple=False).squeeze(1)
            idxs_pe_all = torch.nonzero(mask_pe, as_tuple=False).squeeze(1)

            n_nf = min(self.num_samples,len(idxs_nf_all))
            n_pe = min(self.num_samples,len(idxs_pe_all))

            idxs_nf = idxs_nf_all[torch.randperm(len(idxs_nf_all))[:n_nf]] if len(idxs_nf_all) > 0 else torch.tensor([], dtype=torch.long)
            idxs_pe = idxs_pe_all[torch.randperm(len(idxs_pe_all))[:n_pe]] if len(idxs_pe_all) > 0 else torch.tensor([], dtype=torch.long)
        else:
            N = min(self.num_samples, images.shape[0])
            idxs = torch.randperm(images.shape[0])[:N]

        out, mask = results
        if images.ndim == 3:
            images = images.unsqueeze(1)

        in_C = images.shape[1]
        device = images.device

        mean = torch.tensor(self.mean).view(1, in_C, 1, 1).to(device)
        std = torch.tensor(self.std).view(1, in_C, 1, 1).to(device)

        if self.seperate_disease:
            self.save_viz(n_nf, idxs_nf, images, std, mean, out, mask, step, 'nf')
            self.save_viz(n_pe, idxs_pe, images, std, mean, out, mask, step, 'pe')
        else:
            self.save_viz(N, idxs, images, std, mean, out, mask, step)

    def save_viz(self, N, idxs, images, std, mean, out, mask, step, directory_complement=None):
        fig, axes = plt.subplots(N, 3, figsize=(10, 3 * N))

        for i,idx in enumerate(idxs):
            img = images[idx]
            img_unnorm = (img * std + mean).clip(0, 1)

            img_plot = img_unnorm.squeeze().cpu().numpy()

            pred_mask = out[idx].argmax(dim=0).cpu().numpy()
            true_mask = mask[idx].cpu().permute(1,2,0).numpy()

            if N > 1:
                axes[i, 0].imshow(img_plot, cmap='gray')
                axes[i, 0].set_title(f'Input Image {i}')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(pred_mask, cmap='jet', interpolation='none')
                axes[i, 1].set_title(f'Predicted Mask {i}')
                axes[i, 1].axis('off')

                axes[i, 2].imshow(true_mask, cmap='jet', interpolation='none')
                axes[i, 2].set_title(f'Ground Truth Mask {i}')
                axes[i, 2].axis('off')
            else:
                axes[0].imshow(img_plot, cmap='gray')
                axes[0].set_title(f'Input Image {i}')
                axes[0].axis('off')

                axes[1].imshow(pred_mask, cmap='jet', interpolation='none')
                axes[1].set_title(f'Predicted Mask {i}')
                axes[1].axis('off')

                axes[2].imshow(true_mask, cmap='jet', interpolation='none')
                axes[2].set_title(f'Ground Truth Mask {i}')
                axes[2].axis('off')
        plt.tight_layout()
        if directory_complement is not None:
            directory = os.path.join(self.directory, directory_complement)
        else:
            directory = self.directory
        fig.savefig(os.path.join(self.output_dir, directory, f'step-{step}.png'))
        plt.close(fig)