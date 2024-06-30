from transformers import ViTConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class UniFormerWithProjectionHeadConfig(ViTConfig):    
    def __init__(
            self, 
            projection_size=None, 
            embed_dim=[64, 128, 320, 512],
            image_size=384,
            in_chans=3,
            depth=[5, 8, 20, 7],
            patch_size=[4, 2, 2, 2],
            head_dim=64, 
            mlp_ratio=4, 
            qkv_bias=True, 
            num_classes=1000, 
            qk_scale=None, 
            representation_size=None,
            drop_rate=0.0, 
            drop_path_rate=0.3,
            attn_drop_rate=0.0, 
            conv_stem=False,
            layer_norm_eps=1e-6,
            **kwargs,
        ):
        super().__init__(
            layer_norm_eps=layer_norm_eps, 
            image_size=image_size, 
            qkv_bias=qkv_bias, 
            **kwargs,
        )
        self.projection_size = projection_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.depth = depth
        self.patch_size = patch_size
        self.head_dim = head_dim
        self.mlp_ratio = mlp_ratio 
        self.num_classes = num_classes
        self.qk_scale = qk_scale
        self.representation_size = representation_size
        self.drop_rate = drop_rate 
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rate = attn_drop_rate
        self.conv_stem = conv_stem
