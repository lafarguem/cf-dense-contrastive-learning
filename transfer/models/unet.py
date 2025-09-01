import torch.nn as nn
import segmentation_models_pytorch as smp
from pretraining.utils.distributed import get_world_size

def unet(
        encoder_name: str = "resnet50",
        encoder_weights: str = None,
        in_channels: int = 3,
        out_channels: int = 1,
        decoder_channels=None,
    ):
        encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )
        encoder_channels = encoder.out_channels

        if decoder_channels is None:
            decoder_channels = [
                encoder_channels[-i - 1] // 2 for i in range(len(encoder_channels) - 1)
            ]

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            decoder_channels=decoder_channels
        )

        if get_world_size() > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        return model