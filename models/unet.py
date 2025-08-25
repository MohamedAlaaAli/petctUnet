from models.blocks import *
import torch.nn as nn
import torch.nn.functional as F

# class Identity(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return x



class PETAdapter(nn.Module):
    """Lightweight 1x1x1 adapter to map PET (1 ch) -> C channels at a given scale."""
    def __init__(self, out_ch):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv3d(1, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, pet_patch, target_feat):
        pet_resized = F.interpolate(
            pet_patch, size=target_feat.shape[2:], mode="trilinear", align_corners=False
        )

        return self.adapter(pet_resized)

# class Identity(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return x



class PETAdapter(nn.Module):
    """Lightweight 1x1x1 adapter to map PET (1 ch) -> C channels at a given scale."""
    def __init__(self, out_ch):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv3d(1, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, pet_patch, target_feat):
        pet_resized = F.interpolate(
            pet_patch, size=target_feat.shape[2:], mode="trilinear", align_corners=False
        )

        return self.adapter(pet_resized)

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
                use_att:bool = False,
                use_res = False,
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
                    use_res = self.use_res, 
                    leaky_negative_slope = leaky_negative_slope
                    )
            ]
        )

        self.adapters_pet = nn.ModuleList(
            [
                PETAdapter(chans)
            ]
        )

        self.fusers = nn.ModuleList(
            [
                nn.Conv3d(in_chans, chans, kernel_size=1)
            ]
        )

        if use_att:
            self.down_att_layers = nn.ModuleList([AttentionBlock3D(chans)])

        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock3D(ch, ch * 2, drop_prob, self.use_res, leaky_negative_slope))
            self.adapters_pet.append(PETAdapter(out_ch = ch*2))
            self.fusers.append(nn.Conv3d(ch*4, ch*2, kernel_size=1, padding='same'))
            if use_att:
                self.down_att_layers.append(AttentionBlock3D(ch * 2))
            ch *= 2
        
        self.conv = ConvBlock3D(ch, ch * 2, drop_prob, self.use_res, leaky_negative_slope)
        if use_att:
            self.conv_att = AttentionBlock3D(ch * 2)

        self.cross_atts=nn.ModuleList()

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()

        if use_att:
            self.up_att = nn.ModuleList()

        for _ in range(num_pool_layers):
            self.up_transpose_conv.append(TransposeConvBlock3D(ch * 2, ch, leaky_negative_slope))
            self.up_conv.append(ConvBlock3D(ch * 2, ch, drop_prob, self.use_res, leaky_negative_slope))
            if use_att:
                self.up_att.append(AttentionBlock3D(ch))
                #self.cross_atts.append(CrossAttentionModule(ch, ch))
            ch //= 2

        self.out_conv = nn.Conv3d(ch * 2, self.out_chans, kernel_size=1, stride=1)


    # def forward(self, image: torch.Tensor, text_embedding=None):
    #     """
    #     Args:
    #         image (torch.Tensor): Input tensor of shape (N, in_chans, D, H, W).
    #         text_embedding (torch.Tensor): The Text embedding to ground on.

    #     Returns:
    #         torch.Tensor: Output tensor of shape (N, out_chans, D, H, W).
    #     """
    #     stack = []
    #     output = image

    #     # Downsampling path
    #     for idx, layer in enumerate(self.down_sample_layers):
    #         output = layer(output)
    #         if self.use_att:
    #             output = self.down_att_layers[idx](output)
    #         stack.append(output)
    #         output = F.avg_pool3d(output, kernel_size=2, stride=2)

    #     output = self.conv(output)
    #     if self.use_att:
    #         output = self.conv_att(output)

    #     # Upsampling path
    #     for idx in range(self.num_pool_layers):
    #         downsample = stack.pop()
    #         output = self.up_transpose_conv[idx](output)

    #         # Padding for odd-sized inputs
    #         diff_d = downsample.shape[-3] - output.shape[-3]
    #         diff_h = downsample.shape[-2] - output.shape[-2]
    #         diff_w = downsample.shape[-1] - output.shape[-1]

    #         if diff_d != 0 or diff_h != 0 or diff_w != 0:
    #             output = F.pad(output, [0, diff_w, 0, diff_h, 0, diff_d], mode='reflect')

    #         output = self.cross_atts[idx](text_embedding, downsample) if text_embedding is not None else output
    #         output = torch.cat([output, downsample], dim=1)
    #         output = self.up_conv[idx](output)
    #         if self.use_att:
    #             output = self.up_att[idx](output)

    #     return self.out_conv(output)


    def forward(self, image: torch.Tensor, text_embedding=None):
        """
        Args:
            image (torch.Tensor): Input tensor of shape (N, in_chans, D, H, W).
            text_embedding (torch.Tensor): The Text embedding to ground on.

        Returns:
            torch.Tensor: Output tensor of shape (N, out_chans, D, H, W).
        """
        stack = []
        output = image

        # Downsampling path
        for idx, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            if self.use_att:
                pet_adapted=self.adapters_pet[idx](image[:,1:, :, :, :], output)
                if idx > 0:
                    print(output.shape, pet_adapted.shape)
                    output = self.fusers[idx](torch.cat((output, pet_adapted), dim=1))
                    output = self.down_att_layers[idx](output)
                else:
                    output = self.down_att_layers[idx](output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=2, stride=2)

        # Bottleneck
        output = self.conv(output)
        if self.use_att:
            output = self.conv_att(output)

        # Upsampling path
        for idx in range(self.num_pool_layers):
            skip_connection = stack.pop()
            output = self.up_transpose_conv[idx](output)

            # Handle shape mismatch due to odd input dims
            diff_d = skip_connection.shape[-3] - output.shape[-3]
            diff_h = skip_connection.shape[-2] - output.shape[-2]
            diff_w = skip_connection.shape[-1] - output.shape[-1]
            if diff_d != 0 or diff_h != 0 or diff_w != 0:
                output = F.pad(output, [0, diff_w, 0, diff_h, 0, diff_d], mode='reflect')

            # --- Cross Attention ---
            #if text_embedding is not None:
                #output = self.cross_atts[idx](text_embedding, output)

            # Concatenate with skip connection
            output = torch.cat([output, skip_connection], dim=1)
            output = self.up_conv[idx](output)

            if self.use_att:
                output = self.up_att[idx](output)

        return self.out_conv(output)



import torch
import torch.nn as nn
from torchsummary import summary

def model_summary(model, input_size, device="cuda"):
    """
    Prints summary and number of parameters for a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
        input_size (tuple): Shape of the input (C, H, W) or (C, D, H, W) for 3D.
        device (str): 'cuda' or 'cpu'.
    """
    # Move model to device
    model = model.to(device)
    
    # Model summary
    print("="*50)
    print("📋 Model Summary:")
    summary(model, input_size=input_size, device=device)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("="*50)
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🛠 Trainable parameters: {trainable_params:,}")
    print(f"📦 Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*50)



