import torch
import numpy as np
import matplotlib.pyplot as plt
from pretraining.evaluation.base_evaluation import BaseEvaluator
from matplotlib.colors import ListedColormap, BoundaryNorm
from pretraining.utils.debug import warn_once
import seaborn as sns
import wandb
import cv2
from scipy.linalg import sqrtm
from matplotlib.patches import Ellipse
import os
from collections import defaultdict
from pretraining.utils.distributed import is_main_process

class ComposedReducer:
    def __init__(self, args):
        self.reducers = args
    
    def fit_transform(self, X, y=None):
        reduced = X
        for reducer in self.reducers:
            if callable(getattr(reducer, "fit_transform", None)):
                reduced = reducer.fit_transform(reduced)
            elif callable(getattr(reducer, "fit", None)):
                reduced = reducer.fit(reduced)
            else:
                raise TypeError(f"Reducer {reducer} has neither fit_transform nor fit method")
        return reduced

def ensure_full_containment(points, center, A, safety=1e-6):
    diffs = points - center
    dists = np.einsum('ij,jk,ik->i', diffs, A, diffs)
    max_dist = dists.max()
    if max_dist > 1:
        A = A / (max_dist + safety)
    return A

def plot_ellipse(center, A, ax, edgecolor='r'):
    vals, vecs = np.linalg.eigh(A)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width, height = 2 / np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    ell = Ellipse(xy=center, width=width, height=height, angle=angle,
                  edgecolor=edgecolor, fc='None', lw=2)
    ax.add_patch(ell)

def visualize_embedding_debug(points, normalized, c, A, hue, sat):
    hsv = np.stack([hue, sat, np.full_like(hue, 255)], axis=1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv[np.newaxis, :, :], cv2.COLOR_HSV2RGB)[0]
    rgb_norm = rgb / 255.0

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].scatter(points[:,0], points[:,1], c='blue', s=20)
    plot_ellipse(c, A, axs[0])
    axs[0].set_title("Original Embeddings with MVEE")
    axs[0].axis('equal')
    axs[0].grid(True)

    axs[1].scatter(normalized[:,0], normalized[:,1], c=rgb_norm, s=20)
    circle = plt.Circle((0, 0), 1, color='red', fill=False, lw=2)
    axs[1].add_patch(circle)
    axs[1].set_title("Normalized Embeddings Colored by HSV")
    axs[1].axis('equal')
    axs[1].grid(True)

    return fig

def normalize_points_to_unit_circle(points, center, A):
    A_sqrt = sqrtm(A)  
    diffs = points - center.T
    normalized = (A_sqrt @ diffs.T).T  
    return normalized

def points_to_hue_saturation(normalized_points):
    x = normalized_points[:, 0]
    y = normalized_points[:, 1]
    angles = np.arctan2(y, x)  
    radius = np.sqrt(x**2 + y**2)  

    hue = ((angles + np.pi) / (2 * np.pi)) * 179
    hue = np.clip(hue, 0, 179).astype(np.uint8)

    sat = np.clip(radius * 255, 0, 255).astype(np.uint8)

    return hue, sat

def get_mvee(points, tol=1e-5, max_iter=1000):
    N, d = points.shape
    u = np.ones(N) / N
    err = 1.0
    iter_count = 0

    while err > tol and iter_count < max_iter:
        c = points.T @ u
        
        diffs = points - c.T
        X = (diffs.T * u) @ diffs
        M = np.einsum('ij,jk,ik->i', diffs, np.linalg.inv(X), diffs)  

        j = np.argmax(M)
        max_M = M[j]

        step_size = (max_M - d - 1) / ((d + 1) * (max_M - 1))
        step_size = max(0, min(step_size, 1))  

        new_u = (1 - step_size) * u
        new_u[j] += step_size

        err = np.linalg.norm(new_u - u)
        u = new_u
        iter_count += 1

    c = points.T @ u
    diffs = points - c.T
    X = (diffs.T * u) @ diffs
    A = np.linalg.inv(X) / d  
    A = ensure_full_containment(points, c, A)
    return c, A

def visualize_colorized_image(image_tensor, embeddings, alpha = 0.5):
    
    if embeddings.ndim == 4 and embeddings.shape[1] == 2:
        
        embedding_np = embeddings.permute(0, 2, 3, 1)
    else:
        embedding_np = embeddings

    V, E_H, E_W, _ = embedding_np.shape

    c, A = get_mvee(embedding_np.reshape(-1, 2))
    normalized_embeddings = normalize_points_to_unit_circle(embedding_np.reshape(-1, 2), c, A)
    hue, sat = points_to_hue_saturation(normalized_embeddings)

    hue_grid = hue.reshape(V, E_H, E_W).astype(np.uint8)
    sat_grid = sat.reshape(V, E_H, E_W).astype(np.uint8)

    image_np = image_tensor.permute(0, 2, 3, 1).cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    ellipse = visualize_embedding_debug(embedding_np.reshape(-1, 2), normalized_embeddings, c, A, hue, sat)

    images_bgr = []
    overlays_bgr = []
    aggregated_bgrs = []
    blended_images = []

    for i in range(V):
        img = image_np[i]
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        H, W = image_bgr.shape[:2]

        hue_up = cv2.resize(hue_grid[i], (W, H), interpolation=cv2.INTER_NEAREST)
        sat_up = cv2.resize(sat_grid[i], (W, H), interpolation=cv2.INTER_NEAREST)

        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        overlay_hsv = np.zeros_like(image_hsv)
        overlay_hsv[..., 0] = hue_up
        overlay_hsv[..., 1] = sat_up
        overlay_hsv[..., 2] = 255

        aggregated_hsv = image_hsv.copy()
        aggregated_hsv[..., 0] = hue_up

        overlay_bgr = cv2.cvtColor(overlay_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        aggregated_bgr = cv2.cvtColor(aggregated_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        blended = cv2.addWeighted(image_bgr, 1 - alpha, overlay_bgr, alpha, 0)

        images_bgr.append(image_bgr)
        overlays_bgr.append(overlay_bgr)
        aggregated_bgrs.append(aggregated_bgr)
        blended_images.append(blended)

    return images_bgr, overlays_bgr, aggregated_bgrs, blended_images, ellipse

def visualize_pixel_embeddings(features: torch.Tensor, labels: torch.Tensor, shape, sample_pixels, reducer, random_seed=42, fit_epochs=[], epoch=0):
    N, V, C, H, W = shape
    total_pixels = N * V * H * W
    pixels_per_image = V * H * W

    first_img_start = 0
    first_img_end = pixels_per_image
    first_img_features = features[first_img_start:first_img_end]
    if labels is not None:
        first_img_labels = labels[first_img_start:first_img_end]

    remaining_indices = torch.arange(first_img_end, total_pixels)
    remaining_to_sample = max(0, sample_pixels - pixels_per_image)

    if remaining_to_sample > 0:
        torch.manual_seed(random_seed)
        rand_idx = torch.randperm(len(remaining_indices))[:remaining_to_sample]
        sampled_indices = remaining_indices[rand_idx]

        sampled_features = features[sampled_indices]
        if labels is not None:
            sampled_labels = labels[sampled_indices]
    else:
        sampled_features = features.new_empty((0, C))
        sampled_labels = labels.new_empty((0,), dtype=labels.dtype) if labels is not None else None

    features_subset = torch.cat([first_img_features, sampled_features], dim=0)
    if labels is not None:
        labels_subset = torch.cat([first_img_labels, sampled_labels], dim=0)
    else:
        labels_subset = None

    features_subset = (features_subset - features_subset.mean(dim=0)) / (features_subset.std(dim=0) + 1e-5)

    features_np = features_subset.cpu().numpy()
    labels_np = labels_subset.cpu().numpy() if labels_subset is not None else None

    if callable(getattr(reducer, "transform", None)):
        if epoch == 1 or fit_epochs == [] or epoch in fit_epochs:
            reduced = reducer.fit_transform(features_np)
        else:
            reduced = reducer.transform(features_np)
    else:
        if callable(getattr(reducer, "fit_transform", None)):
            reduced = reducer.fit_transform(features_np)
        elif callable(getattr(reducer, "fit", None)):
            reduced = reducer.fit(features_np)
        else:
            raise TypeError(f"Reducer {reducer} has neither fit_transform nor fit method")

    return reduced, labels_np

class VisualizationEvaluator(BaseEvaluator):
    def __init__(
            self, 
            eval_name,
            eval_freq, 
            reducer,
            all=True,
            one_view=False,
            one_image=False,
            sample_pixels=10000, 
            random_seed=42,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            cf_suffixes = [],
            fit_epochs = [],
            dataset=None):
        super().__init__(eval_name, eval_freq)
        self.epoch = 1
        self.reducer = reducer.reducer
        self.reducer_name = reducer.name
        self.sample_pixels = sample_pixels
        self.random_seed = random_seed
        self.all = all
        self.one_view = one_view
        self.one_image = one_image
        self.mean = mean
        self.std = std
        self.cf_suffixes = cf_suffixes
        self.fit_epochs = fit_epochs

    def evaluate(self, step, encoder, embeddings, labels, shape, logger=None):
        if not is_main_process():
            return None
        reduced, labels_np = visualize_pixel_embeddings(
            embeddings, labels, shape, self.sample_pixels, self.reducer, 
            self.random_seed, self.fit_epochs, self.epoch
        )
        self.epoch += self.eval_freq
        return reduced, labels_np
    
    def _log(self, step, input, results, logger, shape):
        if not is_main_process():
            return None
        reduced, labels_np = results

        N, V, C, H, W = shape

        figs = []
        if labels_np is None:
            labels_one_image = None
            labels_one_view = None
            warn_once(logger, 'pca_no_labels', 'No labels given to PCA visualizer. Classes will not be labeled.')
        else:
            labels_one_image = labels_np[:V*H*W]
            labels_one_view = labels_np[:H*W]

        if self.all:
            os.makedirs(f'{self.output_dir}/visualizations/step-{step}/all')
            fig = self._log_aux(reduced, shape, labels_np, s=2, alpha=0.4)
            fig.savefig(f'{self.output_dir}/visualizations/step-{step}/all/embeddings.png')
            figs.append(fig)
        
        if self.one_image:
            os.makedirs(f'{self.output_dir}/visualizations/step-{step}/one_image')
            fig = self._log_aux(reduced[:V*H*W], shape, labels_one_image, s=10, alpha=0.5)
            fig.savefig(f'{self.output_dir}/visualizations/step-{step}/one_image/embeddings.png')
            figs.append(fig)

            image_tensor = input['x'][0]  
            in_C = image_tensor.shape[1]
            embeddings_tensor = reduced[:V*H*W].reshape(V, H, W, 2)  

            mean = torch.tensor(self.mean).view(1, in_C, 1, 1).to(image_tensor.device)  
            std = torch.tensor(self.std).view(1, in_C, 1, 1).to(image_tensor.device)
            img_unnorm = (image_tensor.unsqueeze(0) * std + mean).squeeze(0).clip(0, 1)

            orig, overlay, aggr, blended, ellipse = visualize_colorized_image(img_unnorm, embeddings_tensor)
            self.log_images(orig, 'one_image/original', step)
            self.log_images(overlay, 'one_image/overlay', step)
            self.log_images(blended, 'one_image/blended', step)
            self.log_images(aggr, 'one_image/aggregated', step)
            ellipse.savefig(os.path.join(self.output_dir, 'visualizations', f'step-{step}', 'one_image', 'color_embeddings.png'))
            plt.close(ellipse)

        if self.one_view:
            os.makedirs(f'{self.output_dir}/visualizations/step-{step}/one_view')
            fig = self._log_aux(reduced[:H*W], shape, labels_one_view, s=10, alpha=0.6)
            fig.savefig(f'{self.output_dir}/visualizations/step-{step}/one_view/embeddings.png')
            figs.append(fig)

            image_tensor = input['x'][0, 0].unsqueeze(0)  
            in_C = image_tensor.shape[1]
            embeddings_tensor = reduced[:H*W].reshape(1, H, W, 2)

            mean = torch.tensor(self.mean).view(1, in_C, 1, 1).to(image_tensor.device)
            std = torch.tensor(self.std).view(1, in_C, 1, 1).to(image_tensor.device)
            img_unnorm = (image_tensor * std + mean).clip(0, 1)

            orig, overlay, aggr, blended, ellipse = visualize_colorized_image(img_unnorm, embeddings_tensor)
            self.log_images(orig, 'one_view/original', step)
            self.log_images(overlay, 'one_view/overlay', step)
            self.log_images(blended, 'one_view/blended', step)
            self.log_images(aggr, 'one_view/aggregated', step)
            ellipse.savefig(os.path.join(self.output_dir, 'visualizations', f'step-{step}', 'one_view', 'color_embeddings.png'))
            plt.close(ellipse)

        logger.info('Logged reduced dim plot')
        for fig in figs:
            plt.close(fig)

    def _log_aux(self, reduced, shape, labels = None, num_views=None, **kwargs):
        if labels is not None:
            unique_labels = np.unique(labels)
            n_classes = len(unique_labels)
            custom_colors = [
                '#e6194b',
                '#3cb44b',
                '#ffe119',
                '#4363d8',
                '#f58231',
                '#911eb4',
            ]

            if n_classes > len(custom_colors):
                raise ValueError(f"Only {len(custom_colors)} custom colors provided for {n_classes} classes.")

            cmap = ListedColormap(custom_colors[:n_classes])
            norm = BoundaryNorm(np.arange(n_classes + 1) - 0.5, n_classes)
            
            fig = plt.figure(figsize=(8, 8))
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap=cmap, norm=norm, **kwargs)

            cbar = plt.colorbar(scatter, ticks=unique_labels)
            cbar.ax.set_yticklabels([str(l) for l in unique_labels])

        else:
            N, V, C, H, W = shape
            points_per_view = H * W

            num_points = reduced.shape[0]

            start_idx = 0

            global_indices = range(start_idx, start_idx + num_points)

            image_indices = []
            view_indices = []

            for idx in global_indices:
                img_idx = idx // (V * points_per_view)
                remainder = idx % (V * points_per_view)
                view_idx = remainder // points_per_view
                
                image_indices.append(img_idx)
                view_indices.append(view_idx)

            unique_images = sorted(set(image_indices))
            unique_views = sorted(set(view_indices))

            markers = ['v', '^', 's', 'D', 'o', '*', 'P', 'X']
            
            max_colors = 50
            num_unique_images = len(unique_images)

            if num_unique_images <= 20:
                base_colors = sns.color_palette("tab20", num_unique_images)
            else:
                base_colors = sns.color_palette("hsv", max(1, min(max_colors, num_unique_images)))

            colors_map = {img: base_colors[i % len(base_colors)] for i, img in enumerate(unique_images)}

            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca()

            grouped_points = defaultdict(list)

            for i, (img, view) in enumerate(zip(image_indices, view_indices)):
                grouped_points[(img, view)].append(i)

            for (img, view), indices in grouped_points.items():
                marker = markers[view % len(markers)]
                color = colors_map[img]
                
                pts = reduced[indices]
                
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    c=[color],
                    marker=marker,
                    **kwargs
                )

        plt.title(f"{self.reducer_name.upper()} of Pixel Embeddings")
        plt.tight_layout()
        return fig
    
    def log_images(self, images, name, step):
        save_dir = os.path.join(self.output_dir, 'visualizations', f'step-{step}', name)
        os.makedirs(save_dir, exist_ok=True)

        for i, img in enumerate(images):
            if i > 0:
                suffix = self.cf_suffixes[i-1] if self.cf_suffixes != [] else i
            else:
                suffix = 'og'
            img_path = os.path.join(save_dir, f'view_{suffix}.png')
            cv2.imwrite(img_path, img)