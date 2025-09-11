import torch
import torch.nn as nn
import torch.nn.functional as F


class UnmixingNetwork(nn.Module):
    def __init__(self, in_channels, num_frames):
        super(UnmixingNetwork, self).__init__()
        self.num_frames = num_frames
        self.feature_extractor = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.frame_decoder = nn.Conv2d(in_channels, num_frames * in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.feature_extractor(x)
        decomposed_frames = self.frame_decoder(features)  # (B, num_frames * C, H, W)
        B, _, H, W = decomposed_frames.size()
        return decomposed_frames.view(B, self.num_frames, -1, H, W)  # (B, num_frames, C, H, W)


class TemporalConvModule(nn.Module):
    def __init__(self, in_channels, num_frames, kernel_size=3, local_size=3):
        super(TemporalConvModule, self).__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.local_size = local_size

        self.depthwise_conv = nn.Conv2d(in_channels*num_frames, in_channels*num_frames, kernel_size, padding=self.padding)

        self.pointwise_conv = nn.Conv2d(in_channels*num_frames, in_channels*num_frames, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W =  x.size()
        x =  x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        x = x.view(B, -1, H, W)  # (B, C * T, H, W)

        temporal_features = self.depthwise_conv(x)

        padded_output = F.pad(temporal_features, [self.local_size // 2] * 4)

        unfold = F.unfold(padded_output, kernel_size=self.local_size)

        unfold = unfold.view(B, C * T, self.local_size * self.local_size, -1)

        mean_values = unfold.mean(dim=2)


        mean_values = mean_values.view(B, C * T, H, W)

        adaptive_weights = mean_values

        weighted_output = temporal_features * adaptive_weights

        pointwise_output = self.pointwise_conv(weighted_output)

        pointwise_output = pointwise_output.view(B, C, T, H, W).permute(0, 2, 1, 3, 4)

        return pointwise_output


class ImageReconstructionNetwork(nn.Module):
    def __init__(self, in_channels, num_frames):
        super(ImageReconstructionNetwork, self).__init__()
        self.reconstructor = nn.Sequential(
            nn.Conv2d(in_channels * num_frames, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, temporal_features):
        B, T, C, H, W = temporal_features.size()

        combined = temporal_features.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        combined = combined.view(B, -1, H, W)  # (B, T * C, H, W)

        reconstructed = self.reconstructor(combined)
        return reconstructed


class TemporalDeblurringNetwork(nn.Module):
    def __init__(self, in_channels, num_frames=9, kernel_size=3):
        super().__init__()
        self.unmixing = UnmixingNetwork(in_channels, num_frames)
        self.temporal_module = TemporalConvModule(in_channels, num_frames, kernel_size)
        self.reconstruction = ImageReconstructionNetwork(in_channels, num_frames)

    def forward(self, x, return_intermediate=False, return_frames=False):
        decomposed_frames = self.unmixing(x)
        if return_frames:
            return decomposed_frames
        temporal_features = self.temporal_module(decomposed_frames)
        output = self.reconstruction(temporal_features)
        if return_intermediate:
            return output, decomposed_frames
        return output

class MSFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFD, self).__init__()
        self.conv1 = BasicConv(in_channels, out_channels - 3, kernel_size = 3,stride=1, relu=True)
        self.downsample_conv2 = BasicConv(out_channels, out_channels * 2 - 3, kernel_size = 3, stride = 2, relu=True)
        self.downsample_conv3 = BasicConv(out_channels * 2, out_channels * 4 - 3, kernel_size = 3, stride = 2, relu=True)

        self.tdn1 = TemporalDeblurringNetwork(in_channels=32)
        self.tdn2 = TemporalDeblurringNetwork(in_channels=64)
        self.tdn3 = TemporalDeblurringNetwork(in_channels=128)

        self.conv21 = BasicConv(out_channels, out_channels, kernel_size=3, stride=1, relu=False)
        self.conv22 = BasicConv(out_channels*2, out_channels*2, kernel_size=3, stride=1, relu=False)
        self.conv23 = BasicConv(out_channels*4, out_channels*4, kernel_size=3, stride=1, relu=False)
    def forward(self, x):
        x_2 = F.interpolate(x, mode='bilinear', align_corners=True)
        x_4 = F.interpolate(x, mode='bilinear', align_corners=True)
        F_1 = torch.cat([x, self.conv1(x)], dim=1)
        F_11, fs1 = self.tdn1(self.conv21(F_1), True)

        F_2 = torch.cat([x_2, self.downsample_conv2(F_1)], dim=1)
        F_21, fs2 = self.tdn2(self.conv22(F_2), True)

        F_3 = torch.cat([x_4, self.downsample_conv3(F_2)], dim=1)
        F_31, fs3 = self.tdn3(self.conv23(F_3), True)

        return F_11, F_21, F_31, fs1, fs2, fs3

class EAGR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EAGR, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

        self.channel_adjust_conv = nn.Conv2d(3, in_channels, kernel_size=1)

        self.edge_conv_x = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.edge_conv_y = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)

        with torch.no_grad():
            self.edge_conv_x.weight[:, :, :, :] = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(1).repeat(1, 3, 1, 1)
            self.edge_conv_y.weight[:, :, :, :] = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(1).repeat(1, 3, 1, 1)

    def forward(self, x, decoupled_features):

        G_x = self.edge_conv_x(x)
        G_y = self.edge_conv_y(x)
        eps = 1e-10
        E_blur = torch.sqrt(G_x * G_x + G_y * G_y + eps)

        E_blur = self.channel_adjust_conv(E_blur)

        A_temp = self.conv1x1(E_blur)
        A_temp = self.bn(A_temp)


        reconstructed_features = decoupled_features * A_temp
        return reconstructed_features

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResBlock1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock1, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class ResBlock2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock2, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class ResBlockStack1(nn.Module):
    def __init__(self, out_channel, num_res=1):
        super(ResBlockStack1, self).__init__()

        layers = [ResBlock1(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ResBlockStack2(nn.Module):
    def __init__(self, out_channel, num_res=1):
        super(ResBlockStack2, self).__init__()

        layers = [ResBlock2(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FeatureFusionNetwork(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusionNetwork, self).__init__()
        self.conv = BasicConv(in_channels, in_channels, kernel_size=3, stride=1)

    def forward(self, feature1, feature2, feature3):
        # Apply channel-level fusion
        combined_features = torch.cat((feature1, feature2, feature3), dim=1)
        fused_output = self.conv(combined_features)
        return fused_output


class LightweightBasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(LightweightBasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = []
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            depthwise_conv = nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding, stride=stride, groups=in_channel, bias=bias)
            pointwise_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias)
            layers.append(depthwise_conv)
            layers.append(pointwise_conv)
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class CombinedAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CombinedAttention, self).__init__()

        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.channel_relu = nn.ReLU()
        self.channel_fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.channel_sigmoid = nn.Sigmoid()

        self.spatial_conv = LightweightBasicConv(in_channels, 1, kernel_size=3, stride=1, relu=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):

        channel_avg_out = self.channel_fc2(self.channel_relu(self.channel_fc1(self.channel_avg_pool(x))))
        channel_max_out = self.channel_fc2(self.channel_relu(self.channel_fc1(self.channel_max_pool(x))))
        channel_weights = self.channel_sigmoid(channel_avg_out + channel_max_out)
        x_channel_att = x * channel_weights

        spatial_weights = self.spatial_sigmoid(self.spatial_conv(x))
        x_spatial_att = x_channel_att * spatial_weights
        return x_spatial_att

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = LightweightBasicConv(channel, channel, kernel_size=3, stride=1, relu=True)
        self.combined_attention = CombinedAttention(channel)

    def forward(self, x1, x2):
        x = x1 * x2
        x = self.merge(x)
        x = self.combined_attention(x)
        out = x1 + x
        return out