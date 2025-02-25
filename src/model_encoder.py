# src/model_encoder.py
"""
PyTorch implementation of the V1 Encoder Model.
This model processes inputs with shape (B, T, C, H, W) where:
    B: Batch size
    T: Number of frames (NUM_FRAMES)
    C: Number of channels (4: raw, opsin, gcamp, orientation)
    H, W: Spatial dimensions (512, 512)
The model applies spatial convolution on each frame (using reshaping to merge B and T),
then uses a ConvLSTM to integrate temporal information, and finally upsamples the latent representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (INPUT_HEIGHT, INPUT_WIDTH, NUM_FRAMES, NUM_CHANNELS,
                    CONV1_FILTERS, CONV1_KERNEL, CONV1_STRIDES,
                    CONV2_FILTERS, CONV2_KERNEL, CONV2_STRIDES,
                    POOL_SIZE, POOL_STRIDES,
                    CONVLSTM1_FILTERS, CONVLSTM1_KERNEL,
                    CONVLSTM2_FILTERS, CONVLSTM2_KERNEL,
                    UP1_FILTERS, UP1_KERNEL, UP1_STRIDES,
                    UP2_FILTERS, UP2_KERNEL, UP2_STRIDES,
                    UP3_FILTERS, UP3_KERNEL, UP3_STRIDES)


# Helper function to convert tuple kernel sizes to integers
def tuple_to_int(t):
    return t if isinstance(t, int) else t[0]


class ConvLSTMCell(nn.Module):
    """A basic ConvLSTM cell."""

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        padding = tuple_to_int(kernel_size) // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state  # h, c: (B, hidden_dim, H, W)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (B, input_dim+hidden_dim, H, W)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):
        height, width = spatial_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """A basic multi-step ConvLSTM module."""

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            self.cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim[i], kernel_size, bias))

    def forward(self, input_tensor):
        # input_tensor shape: (B, T, C, H, W)
        b, t, _, h, w = input_tensor.size()
        hidden_states = []
        layer_output = input_tensor
        for i, cell in enumerate(self.cell_list):
            h_cur, c_cur = cell.init_hidden(b, (h, w))
            outputs = []
            for time_step in range(t):
                h_cur, c_cur = cell(layer_output[:, time_step, :, :, :], (h_cur, c_cur))
                outputs.append(h_cur.unsqueeze(1))
            layer_output = torch.cat(outputs, dim=1)  # (B, T, hidden_dim, H, W)
            hidden_states.append((h_cur, c_cur))
        return layer_output, hidden_states


class V1Encoder(nn.Module):
    def __init__(self):
        super(V1Encoder, self).__init__()
        # Input expected shape: (B, T, C, H, W)
        # Spatial feature extraction: process each time frame with CNN.
        # We'll reshape (B*T, C, H, W), apply CNN layers, then reshape back.
        self.conv1 = nn.Conv2d(NUM_CHANNELS, CONV1_FILTERS, kernel_size=CONV1_KERNEL,
                               stride=CONV1_STRIDES, padding=tuple_to_int(CONV1_KERNEL) // 2)
        self.conv2 = nn.Conv2d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=CONV2_KERNEL,
                               stride=CONV2_STRIDES, padding=tuple_to_int(CONV2_KERNEL) // 2)
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_STRIDES)
        self.conv_complex = nn.Conv2d(CONV2_FILTERS, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_STRIDES)
        self.lateral_module = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # After spatial processing, assume spatial dimensions become:
        # Input 512 -> conv1 (stride2) -> 256, pool1 -> 128, conv_complex -> 128, pool2 -> 64.
        self.spatial_out_dim = 64  # height and width
        # Now define ConvLSTM layers.
        self.convlstm = ConvLSTM(input_dim=128, hidden_dim=[CONVLSTM1_FILTERS, CONVLSTM2_FILTERS],
                                 kernel_size=CONVLSTM1_KERNEL, num_layers=2, bias=True)
        # After ConvLSTM, output is from last layer; assume spatial dims remain 64x64.
        # Upsampling: use ConvTranspose2d to go from 64 -> 128 -> 256 -> 512.
        self.up1 = nn.ConvTranspose2d(CONVLSTM2_FILTERS, UP1_FILTERS, kernel_size=UP1_KERNEL,
                                      stride=UP1_STRIDES, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(UP1_FILTERS, UP2_FILTERS, kernel_size=UP2_KERNEL,
                                      stride=UP2_STRIDES, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(UP2_FILTERS, UP3_FILTERS, kernel_size=UP3_KERNEL,
                                      stride=UP3_STRIDES, padding=1, output_padding=1)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        # Merge batch and time for spatial processing.
        x = x.view(B * T, C, H, W)  # (B*T, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv_complex(x))
        x = self.pool2(x)
        x = F.relu(self.lateral_module(x))
        # x now shape: (B*T, 128, H', W') with H' = W' = 64
        # Reshape back to (B, T, features, H', W')
        _, feat, H_new, W_new = x.size()
        x = x.view(B, T, feat, H_new, W_new)
        # Permute for ConvLSTM: our ConvLSTM expects input of shape (B, T, C, H, W) so it's fine.
        # Pass through ConvLSTM layers
        x, _ = self.convlstm(x)  # x shape: (B, T, hidden, H, W)
        # We'll take the output of the last time step
        x = x[:, -1, :, :, :]  # (B, hidden, H, W), where hidden = CONVLSTM2_FILTERS
        # Upsample to 512x512:
        x = F.relu(self.up1(x))  # output shape: (B, UP1_FILTERS, 128, 128)
        x = F.relu(self.up2(x))  # (B, UP2_FILTERS, 256, 256)
        x = self.up3(x)  # (B, 1, 512, 512)
        return x


if __name__ == "__main__":
    model = V1Encoder()
    print(model)
    # Create dummy input: shape (1, NUM_FRAMES, NUM_CHANNELS, 512, 512)
    dummy_input = torch.zeros(1, NUM_FRAMES, NUM_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    output = model(dummy_input)
    print("Output shape:", output.shape)
