from .attention_blocks import *
import torch
from torch import nn
from torch import nn
import torch.nn.functional as F
from .utils import *
from .archs import *

class UKAN(nn.Module):
    def __init__(self, output_nc=3, input_nc=1, img_size=256, embed_dims=[256, 320, 512] ,drop_rate=0., drop_path_rate=0., kan_norm_layer=nn.LayerNorm, norm_layer=nn.BatchNorm2d, depths=[1, 1, 1], att_type=None,weightshare=False, gpu_ids=[], **kwargs):
        super().__init__()

        self.gpu_ids = gpu_ids
        self.device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")

        in_channels = input_nc 
        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(in_channels, kan_input_dim // 8, norm_layer)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4, norm_layer)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim, norm_layer)

        self.norm3 = kan_norm_layer(embed_dims[1])
        self.norm4 = kan_norm_layer(embed_dims[2])
        self.dnorm3 = kan_norm_layer(embed_dims[1])
        self.dnorm4 = kan_norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([KANBlock(embed_dims[1],no_kan=False, drop=drop_rate, drop_path=dpr[0], norm_layer=kan_norm_layer)])
        self.block2 = nn.ModuleList([KANBlock(embed_dims[2],no_kan=False,  drop=drop_rate, drop_path=dpr[1], norm_layer=kan_norm_layer)])
        self.dblock1 = nn.ModuleList([KANBlock(embed_dims[1],no_kan=False,  drop=drop_rate, drop_path=dpr[0], norm_layer=kan_norm_layer)])
        self.dblock2 = nn.ModuleList([KANBlock(embed_dims[0], no_kan=False, drop=drop_rate, drop_path=dpr[1], norm_layer=kan_norm_layer)])

        # self.patch_embed3 = PatchEmbed(img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        # self.patch_embed4 = PatchEmbed(img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.patch_embed3 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.weightshare = weightshare
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1], norm_layer)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8, norm_layer)

        if not weightshare:
            self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0], norm_layer)
            self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4, norm_layer)
            self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8, norm_layer)
        else:
            self.decoder2 = self.encoder3  # share weights symmetrically
            self.decoder3 = self.encoder2  # weight sharing with encoder2
            self.decoder4 = self.encoder1



        if att_type == 'SENet':
            Att = SE
        elif att_type == 'ChannelSpatialSELayer':
            Att = ChannelSpatialSELayer
       
        else:
            Att = nn.Identity

        self.attn_d2 = Att(embed_dims[0])
  
        self.attn_d4 = Att(embed_dims[0] // 8)

      
        self.final = nn.Conv2d(embed_dims[0] // 8, output_nc, kernel_size=1)
       

        # Learnable skip connection weights
        self.skip_weight1 = nn.Parameter(torch.tensor(1.0))  # for t1
        self.skip_weight2 = nn.Parameter(torch.tensor(1.0))  # for t2
        self.skip_weight3 = nn.Parameter(torch.tensor(1.0))  # for t3
        self.skip_weight4 = nn.Parameter(torch.tensor(1.0))  # for t4



    def forward(self, x):
        B = x.shape[0]

 # encoder blocks

        # Encoder block 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out

        # Encoder block 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out

        # Encoder block 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

 # patch embed and KAN blocks

        out, H, W = self.patch_embed3(out)
        for blk in self.block1: 
            out = blk(out, H, W)
        out = self.norm3(out).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, H, W = self.patch_embed4(out)
        for blk in self.block2: 
            out = blk(out, H, W)
        out = self.norm4(out).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

# decoder blocks

        # decoder block 1
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear'))
        if t4.shape[2:] != out.shape[2:]:
            t4 = F.interpolate(t4, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = torch.add(out, self.skip_weight4 * t4)

        # kan block in decoder block 1
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # decoder block 2
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.attn_d2(out)
        if t3.shape[2:] != out.shape[2:]:
            t3 = F.interpolate(t3, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = torch.add(out, self.skip_weight3 * t3)

        # kan block in decoder block 2
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # decoder block 3
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))

        if t2.shape[2:] != out.shape[2:]:
            t2 = F.interpolate(t2, size=out.shape[2:], mode='bilinear', align_corners=False)

        out = torch.add(out, self.skip_weight2 * t2)
    
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.attn_d4(out)
        
        if t1.shape[2:] != out.shape[2:]:
            t1 = F.interpolate(t1, size=out.shape[2:], mode='bilinear', align_corners=False)

        out = torch.add(out, self.skip_weight1 * t1)
        
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear'))
      
        return torch.tanh(self.final(out))


