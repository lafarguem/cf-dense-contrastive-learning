from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
from pretraining.evaluation.base_evaluation import BaseEvaluator
import wandb
from pretraining.utils.distributed import is_main_process

def cluster_and_vote(features, labels, num_clusters):
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    kmeans = KMeans(n_clusters=num_clusters, n_init=20)
    cluster_ids = kmeans.fit_predict(features_np)

    cluster_to_label = {}
    for c in range(num_clusters):
        idxs = (cluster_ids == c)
        if np.sum(idxs) == 0:
            cluster_to_label[c] = -1
        else:
            majority_label = np.bincount(labels_np[idxs]).argmax()
            cluster_to_label[c] = majority_label

    pred_labels = np.array([cluster_to_label[c] for c in cluster_ids])
    acc = accuracy_score(labels_np, pred_labels) * 100

    return acc, cluster_ids, pred_labels


class KMeansEvaluator(BaseEvaluator):
    def __init__(self, eval_name, eval_freq, num_classes, dataset=None):
        super().__init__(eval_name, eval_freq)
        self.num_classes = num_classes
    
    def evaluate(self, step, encoder, embeddings, labels=None, shape=None, logger=None):
        if not is_main_process():
            return
        if labels is None:
            return None
        acc, _, _ = cluster_and_vote(embeddings, labels, num_clusters=self.num_classes)
        return acc
    
    def _log(self, step, input, results, logger, shape):
        if not is_main_process():
            return
        wandb.log({f"eval/{self.eval_name}": results}, step=step)
        logger.info(f'K-means accuracy {results:.4f}')
        return {'k_means_accuracy': (results, -1)}