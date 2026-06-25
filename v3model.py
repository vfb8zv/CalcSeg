import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from medsam import MedSAMEncoder


class SliceDAttention(nn.Module):
    def __init__(self, in_channels=64, embed_dim=None):
        super().__init__()

        self.embed_dim = embed_dim or in_channels
        self.q = nn.Linear(in_channels, self.embed_dim)
        self.k = nn.Linear(in_channels, self.embed_dim)
        self.v = nn.Linear(in_channels, self.embed_dim)
        self.out = nn.Linear(self.embed_dim, in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.register_buffer(
            "slice_pos_emb",
            self._build_sincos_pe(max_len=12, dim=in_channels))
        
    def _build_sincos_pe(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, mask=None, alpha_=1.0):
        B, C, F, H, W = x.shape
        x_flat = x.mean(dim=[3, 4]).permute(0,2,1)  
        x_flat = x_flat + self.slice_pos_emb.unsqueeze(0)
        
        Q = self.q(x_flat)
        K = self.k(x_flat)
        V = self.v(x_flat)

        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (C ** 0.5) 
        if mask is not None:
            mask = mask.unsqueeze(1)  
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        
        attn = torch.softmax(attn_scores, dim=-1) 
        out = torch.matmul(attn, V)  
        out = self.out(out) 
        out = self.norm(out)
        out = out.permute(0, 2, 1)            
        out = out.unsqueeze(-1).unsqueeze(-1) 
        out = out.expand(-1, -1, -1, H, W)
        out = out + alpha_*x
        return out


class ResidualConvBlock(nn.Module):
    """
    Residual conv block: reduces overfitting and stabilizes gradients.
    """
    def __init__(self, in_ch, out_ch):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.relu(x)
        return x


class CalcSeg(nn.Module):
    
    def __init__(self, pretrained_path='./results/medsam_vit_b.pth', 
                 num_classes=1, freeze_encoder=False, dropout_p=0.3, fp_float=None):
        
        super(CalcSeg, self).__init__()

        self.medsam_encoder = MedSAMEncoder(
            pretrained_path=pretrained_path,
            freeze_encoder=freeze_encoder)

        self.project = nn.Conv2d(224, 64, kernel_size=1)
        
        self.scar_block1 = ResidualConvBlock(64, 64)
        self.scar_block2 = ResidualConvBlock(64, 32)

        if fp_float is not None:
            self.lambda_neg = nn.Parameter(torch.tensor(fp_float))
        else:
            self.lambda_neg = None

        self.st_attention_1h = SliceDAttention()

        self.final_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(32, 1, kernel_size=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None, Fr=12):
        B_, C, H, W = x.shape
        B = B_//Fr
        
        if mask is not None:
            fused = torch.zeros(B*Fr, 64, H, W, device=x.device)
            mask_flat = mask.reshape(B * Fr)
            valid_idx = mask_flat.nonzero(as_tuple=True)[0]
            
            med_features_valid = self.medsam_encoder(x[valid_idx])
            fused_valid = torch.cat(med_features_valid, dim=1)
            fused[valid_idx] = self.project(fused_valid)
        else:
            med_features = self.medsam_encoder(x)
            fused = torch.cat(med_features, dim=1)
            fused = self.project(fused)
            
        fused = fused.reshape(B, Fr, 64, fused.shape[-2], fused.shape[-1])
        fused = fused.permute(0, 2, 1, 3, 4) #bcfhw
        fused = F.interpolate(fused, size=(Fr, 64, 64), mode='trilinear', align_corners=False)
        fused = self.st_attention_1h(fused, mask)
        fused = F.interpolate(fused, size=(Fr, 224, 224), mode='trilinear', align_corners=False)
        fused = fused.permute(0, 2, 1, 3, 4) #bfchw
        fused = fused.reshape(B * Fr, 64, fused.shape[-2], fused.shape[-1])

        scar_feat = self.scar_block1(fused)
        scar_feat = self.scar_block2(scar_feat)
        
        fused_out = scar_feat
        out = self.final_head(fused_out)

        return out
