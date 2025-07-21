import torch.nn as nn
import torch


class ConvBlock3D(nn.Module):
    """
        A 3D Convolutional Block consisting of two convolutional layers, each followed by
        instance normalization, LeakyReLU activation, and dropout. Optionally includes a
        residual connection using a 1x1 convolution.

    """

    def __init__(self, in_chans:int, out_chans:int, drop_prob:float, use_res:bool=True, leaky_negative_slope:float=0.0):
        """
            Args:
                in_chans (int): Number of input channels.
                out_chans (int): Number of output channels.
                drop_prob (float): Dropout probability.
                use_res (bool): Whether to use a residual connection. Default is True.
                leaky_negative_slope (float): Negative slope for the LeakyReLU activation. Default is 0.
        """
        super().__init__()
        self.use_res = use_res
        self.leaky_negative_slope = leaky_negative_slope

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(leaky_negative_slope, inplace=True),
            nn.Dropout3d(drop_prob),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.out_layers = nn.Sequential(
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(leaky_negative_slope, inplace=True),
            nn.Dropout3d(drop_prob),
        )

    def forward(self, x:torch.Tensor):
        """
            Args:
                x (torch.Tensor): Input tensor of shape (N, C, D, H, W).

            Returns:
                torch.Tensor: Output tensor of shape (N, out_chans, D, H, W).
        """
        if self.use_res:
            return self.out_layers(self.layers(x) + self.conv1x1(x))
        else:
            return self.out_layers(self.layers(x))


class TransposeConvBlock3D(nn.Module):
    """
        A 3D transpose convolutional block used for upsampling in volumetric data, consisting of a 
        3D transpose convolution (deconvolution) that doubles the spatial dimensions, Instance normalization and 
        LeakyReLU activation.

    """

    def __init__(self, in_chans:int, out_chans:int, leaky_negative_slope:float=0.0):
        """        
            Args:
                in_chans (int): Number of input channels.
                out_chans (int): Number of output channels.
                leaky_negative_slope (float): Negative slope for the LeakyReLU activation function. Default is 0.0.
        """
        super().__init__()
        self.leaky_negative_slope = leaky_negative_slope
        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(leaky_negative_slope, inplace=True),
        )

    def forward(self, x:torch.Tensor):
        """
            Args:
                x (torch.Tensor): Input tensor of shape (N, in_chans, D, H, W).

            Returns:
                torch.Tensor: Upsampled tensor of shape (N, out_chans, 2*D, 2*H, 2*W).
        """
        return self.layers(x)


class AttentionBlock3D(nn.Module):
    """
        A 3D attention module that applies both channel and spatial attention mechanisms.

        The block uses a global pooling operation followed by a lightweight two-layer MLP 
        to compute channel attention. For spatial attention, it applies a 1x1x1 convolution 
        to generate a spatial attention map. The final output is the element-wise maximum 
        between the channel-attended and spatial-attended feature maps.

    """
    def __init__(self, num_chans, reduction=2):
        """       
            Args:
                num_chans (int): The number of input channels.
                reduction (int): The reduction ratio used in the channel attention bottleneck. Defaults to 2.
        """
        super().__init__()
        self.C = num_chans
        self.r = reduction

        self.sig = nn.Sigmoid()
        # Channel attention
        self.fc_ch = nn.Sequential(
            nn.Linear(self.C, self.C // self.r),
            nn.ReLU(inplace=True),
            nn.Linear(self.C // self.r, self.C),
        )
        # Spatial attention
        self.conv = nn.Conv3d(self.C, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):  
        """
            Args:
                x (torch.Tensor): Input tensor of shape (N, in_chans, D, H, W).

            Returns:
                torch.Tensor: Output tensor of shape (N, in_chans, D, H, W), element-wise maximum between the channel-attended and spatial-attended feature maps.

        """
        b, c, d, h, w = x.shape

        # Spatial attention
        sa = self.sig(self.conv(x))
        x_sa = sa * x

        # Channel attention
        ca = torch.mean(torch.abs(x).reshape(b, c, -1), dim=2)
        ca = self.sig(self.fc_ch(ca)).reshape(b, c, 1, 1, 1)
        x_ca = ca * x

        return torch.max(x_sa, x_ca)


class CrossAttentionModule(nn.Module):
    def __init__(self, in_dim_q, embed_dim, in_dim_kv=768,  n_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(in_dim_q, embed_dim)
        self.k_proj = nn.Linear(in_dim_kv, embed_dim)
        self.v_proj = nn.Linear(in_dim_kv, embed_dim)

        self.att = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        # Optional: fuse with original image features
        self.fuse = nn.Tanh()

    def forward(self, text_emb, img_feats):
        """
        Args:
            text_emb: [B, L_text, in_dim_q] (RadBERT output after linear layer)
            img_feats: [B, C_img, D, H, W] (feature map from UNet)
        Returns:
            Attended features: [B, C_img, D, H, W]
        """

        B, C_img, D, H, W = img_feats.shape
        N = D * H * W

        img_feats_flat = img_feats.flatten(2).transpose(1, 2)
        # Linear projections
        Q = self.q_proj(img_feats_flat)          # [B, N, embed_dim]
        K = self.k_proj(text_emb)                # [B, L_text, embed_dim]
        V = self.v_proj(text_emb)                # [B, L_text, embed_dim]
        # QKT = B, N, Ltxt --> rsltV --> B, N, embed_dim 
        # Cross-attention: attend text over image
        attn_output, _ = self.att(Q, K, V)        # [B, N, embed_dim]

        # Fuse text-image output
        attn_output = self.fuse(attn_output)      # [B, N, embed_dim]
        print(attn_output.shape)

        attn_output = attn_output.permute(0, 2, 1).view(B, C_img, D, H, W)
        fused_out = img_feats * attn_output  
        return fused_out

