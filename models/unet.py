from blocks import *
import torch.nn as nn

class Unet(nn.Module):
    """
        Implements a 3D Unet architecture.
    """
    def __init__(self,
                in_chans:int = 1,
                out_chans:int = 1,
                chans:int = 32,
                num_pool_layers:int = 4,
                drop_prob:float = 0.2,
                use_res:bool = False,
                use_att:bool = False, 
                leaky_negative_slope:float = 0.0):
        """
            Args:
                in_chans (int): Number of input channels. Default is 1.
                out_chans (int): Number of output channels. Default is 1.
                chans (int): Number of intermediate channels. Default is 32.
                drop_prob (float): Dropout probability. Default is 0.2.
                use_res (bool): Whether to use a residual connection. Default is False.
                use_att (bool): Whether to use spatial and channel attentions. Default is False
                leaky_negative_slope (float): Negative slope for the LeakyReLU activation. Default is 0.

        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.use_res = use_res
        self.use_att = use_att
        self.leaky_negative_slope = leaky_negative_slope

        self.down_sample_layers = nn.ModuleList(
            [
                ConvBlock3D(
                    in_chans = in_chans,
                    out_chans = chans,
                    drop_prob = drop_prob, 
                    use_res = use_res, 
                    leaky_negative_slope = leaky_negative_slope
                    )
            ]
        )

        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock3D(ch, ch * 2, drop_prob, use_res, leaky_negative_slope))
            if use_att:
                self.down_att_layers.append(AttentionBlock3D(ch * 2))
            ch *= 2
        
        self.conv = ConvBlock3D(ch, ch * 2, drop_prob, use_res, leaky_negative_slope)
        if use_att:
            self.conv_att = AttentionBlock3D(ch * 2)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()

        if use_att:
            self.up_att = nn.ModuleList()

        for _ in range(num_pool_layers):
            self.up_transpose_conv.append(TransposeConvBlock3D(ch * 2, ch, leaky_negative_slope))
            self.up_conv.append(ConvBlock3D(ch * 2, ch, drop_prob, use_res, leaky_negative_slope))
            if use_att:
                self.up_att.append(AttentionBlock3D(ch))
            ch //= 2

        self.out_conv = nn.Conv3d(ch * 2, self.out_chans, kernel_size=1, stride=1)




