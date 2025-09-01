import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from pretraining.models.contrastive.base_model import SingleBranchContrastiveModel
from pretraining.utils.distributed import get_world_size

class Unet(SingleBranchContrastiveModel):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = None,
        in_channels: int = 3,
        out_channels: int = 1,
        decoder_channels=None,
        has_decoder=True,
        return_global=False,
        return_dense=True,
        runtime_args=None,
        disable_instance=False,
        dense_projection_head=[],
        instance_projection_head=[],
    ):
        super().__init__(dense_projection_head,instance_projection_head,disable_instance,
                         has_decoder,return_global,return_dense,runtime_args)

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

        self.encoder = model.encoder
        if self.has_decoder:
            self.decoder = model.decoder
            self.segmentation_head = model.segmentation_head

        if self.has_decoder:
            self.out_channels = decoder_channels[-1]
            self.encoder_out_channels = encoder_channels[-1]
        else:
            self.out_channels = encoder_channels[-1]
            self.encoder_out_channels = self.out_channels
        self.setup_head()
        del model, encoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, V, C, H, W]
        Returns: Tensor of shape [B, V, C1, H2, W2]
        """
        x = x['x']
        if x.ndim == 5:
            B, V, C, H, W = x.shape
            x = x.view(B * V, C, H, W)
            features = self.encoder(x)
            deepest = features[-1]
            _, C1, H2, W2 = deepest.shape

            return features, deepest.view(B, V, C1, H2, W2)
        elif x.ndim == 4:
            features = self.encoder(x)
            deepest = features[-1]
            _, C1, H2, W2 = deepest.shape

            return features, deepest
        else:
            raise ValueError(f'Invalid input. x should be of shape [B,V,C,H,W] or [B,C,H,W]. Got {x.shape}')

    def decode(self, encoding) -> torch.Tensor:
        """
        features: Tensor of shape [B, V, C1, H2, W2]
        Returns: Tensor of shape [B, V, C2, H, W]
        """
        features, deepest = encoding
        if deepest.ndim == 5:
            B,V = deepest.shape[0], deepest.shape[1]
            decoded = self.decoder(*features)
            C2, H, W = decoded.shape[1:]
            return decoded.view(B, V, C2, H, W)
        elif deepest.ndim == 4:
            decoded = self.decoder(*features)
            return decoded
    
    def encoded_for_global(self, encoding):
        _, embeddings = encoding
        return embeddings
    
    def encoded_for_dense(self, encoding):
        return self.encoded_for_global(encoding)
    
    def get_pretrained_model(self):
        state = {'encoder': self.encoder.state_dict()}
        if self.has_decoder:
            state['decoder'] = self.decoder.state_dict()
        return state