import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer for channel attention.
    Paper: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)
        
class ScarAttention(nn.Module):
    """
    Specialized attention mechanism for scar regions in cardiac MRI.
    Uses spatial and channel attention to focus on scar-relevant features.
    """
    def __init__(self, in_channels):
        super(ScarAttention, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply channel and spatial attention to input features."""
        # Channel attention
        channel_att = self.channel_gate(F.adaptive_avg_pool2d(x, 1))
        x = x * channel_att
        
        # Spatial attention
        spatial_att = self.spatial_gate(x)
        x = x * spatial_att
        
        return x

class MedSAMEncoder(nn.Module):
    """MedSAM-based encoder with enhanced attention mechanisms."""
    
    def __init__(self, pretrained_path, freeze_encoder=False):
        super(MedSAMEncoder, self).__init__()
        
        # Initialize MedSAM
        self.model_type = "vit_b"
        sam = sam_model_registry[self.model_type]()
        self.image_encoder = sam.image_encoder
    
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            encoder_state_dict = {
                k.replace('image_encoder.', ''): v
                for k, v in state_dict.items()
                if k.startswith('image_encoder.')}
        
            missing, unexpected = self.image_encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"Loaded MedSAM encoder weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")
        
        # Additional processing layers
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                SELayer(128),
                ScarAttention(128)
            ),
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                SELayer(64),
                ScarAttention(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                SELayer(32),
                ScarAttention(32)
            )
        ])
        
        self.upsampler = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def interpolate_pos_encoding(self, x, pos_embed):
        """Interpolate positional encoding to match input size."""
        N, H, W, C = x.shape
        if len(pos_embed.shape) == 3:
            pos_embed = pos_embed.reshape(1, int(pow(pos_embed.shape[1], 0.5)), 
                                       int(pow(pos_embed.shape[1], 0.5)), C)
        pos_embed = nn.functional.interpolate(
            pos_embed.permute(0, 3, 1, 2),
            size=(H, W),
            mode='bicubic',
            align_corners=False
        ).permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x):
        # Handle grayscale input
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Patch embedding
        x = self.image_encoder.patch_embed(x)
        
        # Add positional encoding
        pos_embed = self.interpolate_pos_encoding(x, self.image_encoder.pos_embed)
        x = x + pos_embed
        
        # Process through transformer blocks
        for blk in self.image_encoder.blocks:
            x = blk(x)
        
        # Process through neck
        x = self.image_encoder.neck(x.permute(0, 3, 1, 2))
        
        # Process through decoder blocks
        features = []
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
            features.append(self.upsampler(x))
        
        return features