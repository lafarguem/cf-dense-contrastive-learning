from monai.metrics import SurfaceDistanceMetric, HausdorffDistanceMetric, DiceMetric
from monai.networks.utils import one_hot

class SegmentationMetricsEvaluator:
    def __init__(self, hausdorff=True, hausdorff_95=True, dice=True, average_surface_distance=True):
        self.metrics = []
        if hausdorff:
            self.metrics.append(('Hausdorff', HausdorffDistanceMetric(), 1))
        if hausdorff_95:
            self.metrics.append(('Hausdorff 95', HausdorffDistanceMetric(percentile=95), 1))
        if dice:
            self.metrics.append(('DICE', DiceMetric(include_background=False, num_classes=3), -1))
        if average_surface_distance:
            self.metrics.append(('ASD', SurfaceDistanceMetric(symmetric=True), 1))

    def __call__(self, y_pred, y):
        y_pred_one_hot = one_hot(y_pred.argmax(dim=1, keepdim=True), num_classes=3, dim=1)
        y_one_hot = one_hot(y.unsqueeze(1), num_classes=3, dim=1)
        output = {}
        for metric in self.metrics:
            metric[1].reset()
            metric[1](y_pred_one_hot, y_one_hot)
            result = metric[1].aggregate()
            output[metric[0]] = (result, metric[2])
            metric[1].reset()
        return output