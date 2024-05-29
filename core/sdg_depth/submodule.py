from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class Feature_ddc(nn.Module):
    def __init__(self, in_plans=32, args=None):
        super(Feature_ddc, self).__init__()
        self.args = args

        self.conv_start_img = nn.Sequential(BasicConv(3, 16, kernel_size=3, padding=1),
                                            BasicConv(16, 16, kernel_size=3, padding=1))

        self.conv_x = nn.Conv2d(in_plans, 16, (3, 3), (1, 1), (1, 1), bias=False)  # warp error conv
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=True))

        self.conv_disp = nn.Conv2d(1, 8, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn_relu_disp = nn.Sequential(nn.BatchNorm2d(8),
                                          nn.ReLU(inplace=True))

        self.conv_depth = nn.Conv2d(1, 8, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn_relu_depth = nn.Sequential(nn.BatchNorm2d(8),
                                           nn.ReLU(inplace=True))

        self.conv_start = nn.Sequential(
            BasicConv(48, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=2, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv0a = BasicConv(48, 32, kernel_size=3, stride=1, padding=1)

        self.conv1a = BasicConv(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(48, 32, deconv=True)
        self.deconv3a = Conv2x(32, 32, deconv=True)
        self.deconv2a = Conv2x(32, 32, deconv=True)
        self.deconv1a = Conv2x(32, 32, deconv=True)

        self.conv1b = Conv2x(32, 32)
        self.conv2b = Conv2x(32, 32)
        self.conv3b = Conv2x(32, 32)
        self.conv4b = Conv2x(32, 48)

        self.deconv4b = Conv2x(48, 32, deconv=True)
        self.deconv3b = Conv2x(32, 32, deconv=True)
        self.deconv2b = Conv2x(32, 32, deconv=True)

        self.deconv1b = Conv2x(32, 32, deconv=True)
        self.deconv0b = Conv2x(32, 32, deconv=True)

        self.conv_out1 = BasicConv(32, 32, kernel_size=3, padding=1)

        self.conv_out2 = BasicConv(64, 32, kernel_size=3, padding=1)

        self.conv_disp_out = BasicConv(40, 1, kernel_size=3, padding=1, relu=False)
        self.conv_depth_out = BasicConv(40, 1, kernel_size=3, padding=1, relu=False)

    def forward(self, x, img=None, disp=None, depth=None, resolution=1):
        x_ori = x

        # resolution = 4
        if resolution != 1:
            img = F.interpolate(img, [img.shape[-2] // resolution, img.shape[-1] // resolution],
                                mode='bilinear', align_corners=False)

        g = self.conv_start_img(img)

        x1 = self.conv_x(x)
        if g.shape[-1] // x1.shape[-1] != 1:
            factor = g.shape[-1] / x1.shape[-1]
            x1 = F.interpolate(x1, [int(x1.size()[2] * factor), int(x1.size()[3] * factor)], mode='bilinear',
                               align_corners=False)
        x1 = self.bn_relu(x1)

        x_disp = self.conv_disp(disp)
        x_disp = self.bn_relu_disp(x_disp)

        x_depth = self.conv_depth(depth)
        x_depth = self.bn_relu_disp(x_depth)

        if x_disp.shape[-1] != x1.shape[-1]:
            x_disp = F.interpolate(x_disp, [x1.shape[-2], x1.shape[-1]], mode='bilinear', align_corners=True)
            x_depth = F.interpolate(x_depth, [x1.shape[-2], x1.shape[-1]], mode='bilinear', align_corners=True)

        x = torch.cat([g, x1, x_disp, x_depth], dim=1)

        rem_0_0 = self.conv0a(x)

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        #
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)

        #
        rem3 = x
        x = self.conv4b(x, rem4)

        #
        x = self.deconv4b(x, rem3)

        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        x = self.deconv0b(x, rem_0_0)

        x = self.conv_out1(x)
        rem0 = F.interpolate(rem0, [x.shape[2], x.shape[3]], mode='bilinear', align_corners=False)
        x = torch.cat([x, rem0], dim=1)
        x = self.conv_out2(x)

        res_disp = self.conv_disp_out(torch.cat([x, x_disp], dim=1))
        res_depth = self.conv_depth_out(torch.cat([x, x_depth], dim=1))

        if resolution != 1:
            res_disp = F.interpolate(res_disp, scale_factor=resolution, mode='bilinear', align_corners=True)
            res_depth = F.interpolate(res_depth, scale_factor=resolution, mode='bilinear', align_corners=True)

        res_disp = (torch.sigmoid(res_disp) - 0.5) * 2.0 * self.args.disp_to_depth_convert_disp_range  # 0.3
        res_depth = (torch.sigmoid(res_depth) - 0.5) * 2.0 * self.args.disp_to_depth_convert_depth_range  # 1.2

        return res_disp, res_depth


def coords_grid(batch, ht, wd, flow=None):
    coords = torch.meshgrid(torch.arange(ht, device=flow.device), torch.arange(wd, device=flow.device))

    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    if sample_coords.size(1) != 2:
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def bilinear_sample_v2(img, coords, mode='bilinear', mask=False, padding_mode='zeros'):  # zeros,border
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    if mode == 'bilinear':
        img = F.grid_sample(img, grid, align_corners=True, padding_mode=padding_mode)
    elif mode == 'nearest':
        img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()

    grid = coords_grid(b, h, w, flow) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


class DDC_Module(nn.Module):
    def __init__(self, in_plans=3, args=None):
        super(DDC_Module, self).__init__()
        self.args = args

        self.refine_net = Feature_ddc(in_plans=3, args=args)

    def forward(self, left, right, disp, depth, conversion_rate, resolution=1):
        flow = torch.cat([disp, torch.zeros_like(disp, device=disp.device)], dim=1)
        right_warp = flow_warp(right, flow, padding_mode='zeros')
        error = left - right_warp
        depth_ori = depth

        res_disp, res_depth = self.refine_net(error, left, disp=disp, depth=depth, resolution=resolution)
        disp = disp + res_disp
        disp = torch.clamp(disp, min=0, max=self.args.max_disp)
        c_rate = conversion_rate.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, disp.shape[-2], disp.shape[-1])
        if self.args.disp_to_depth_convert_gate == 3.8:
            mask = disp > self.args.disp_to_depth_convert_gate
            depth = c_rate / (disp + 1e-6)
            depth = torch.where(mask, depth, depth_ori)

        else:
            mask = disp > self.args.disp_to_depth_convert_gate
            depth = c_rate / (disp + 1e-6)

        depth = torch.clamp(depth, min=0, max=100.)

        return disp, depth


class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes, model_name='pspnet', fusion_mode='cat', with_bn=True):
        super(pyramidPooling, self).__init__()

        bias = not with_bn

        self.paths = []
        if pool_sizes is None:
            for i in range(4):
                self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, bias=bias, with_bn=with_bn))
        else:
            for i in range(len(pool_sizes)):
                self.paths.append(
                    conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias,
                                        with_bn=with_bn))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    # @profile
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        if self.pool_sizes is None:
            for pool_size in np.linspace(2, min(h, w), 4, dtype=int):
                k_sizes.append((int(h / pool_size), int(w / pool_size)))
                strides.append((int(h / pool_size), int(w / pool_size)))
            k_sizes = k_sizes[::-1]
            strides = strides[::-1]
        else:
            k_sizes = [(self.pool_sizes[0], self.pool_sizes[0]), (self.pool_sizes[1], self.pool_sizes[1]),
                       (self.pool_sizes[2], self.pool_sizes[2]), (self.pool_sizes[3], self.pool_sizes[3])]
            strides = k_sizes

        if self.fusion_mode == 'cat':  # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.upsample(out, size=(h, w), mode='bilinear')
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else:  # icnet: element-wise sum (including x)
            pp_sum = x

            for i, module in enumerate(self.path_module_list):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                out = module(out)
                out = F.upsample(out, size=(h, w), mode='bilinear')
                pp_sum = pp_sum + 0.25 * out
            pp_sum = F.relu(pp_sum / 2., inplace=True)
            # pp_sum = FMish(pp_sum / 2.)

            return pp_sum


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")

    def forward(self, x):
        # save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x * (torch.tanh(F.softplus(x)))


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def disparity_variance(x, maxdisp, disparity):
    # the shape of disparity should be B,1,H,W, return is the variance of the cost volume [B,1,H,W]
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    disp_values = (disp_values - disparity) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)


def disparity_variance_confidence(x, disparity_samples, disparity):
    # the shape of disparity should be B,1,H,W, return is the uncertainty estimation
    assert len(x.shape) == 4
    disp_values = (disparity - disparity_samples) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def groupwise_correlation_4D(fea1, fea2, num_groups):
    B, C, D, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, D, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, D, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def FMish(x):
    '''

    Applies the mish function element-wise:

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    See additional documentation for mish class.

    '''

    return x * torch.tanh(F.softplus(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.gc(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class UniformSampler(nn.Module):
    def __init__(self):
        super(UniformSampler, self).__init__()

    def forward(self, min_disparity, max_disparity, number_of_samples=10, offset=None):
        """
        Args:
            :min_disparity: lower bound of disparity search range
            :max_disparity: upper bound of disparity range predictor
            :number_of_samples (default:10): number of samples to be genearted.
        Returns:
            :sampled_disparities: Uniformly generated disparity samples from the input search range.
        """

        device = min_disparity.get_device()

        multiplier = (max_disparity - min_disparity) / (number_of_samples + 1)  # B,1,H,W
        range_multiplier = torch.arange(1.0, number_of_samples + 1, 1, device=device).view(number_of_samples, 1,
                                                                                           1)  # (number_of_samples, 1, 1)
        if offset is None:
            sampled_disparities = min_disparity + multiplier * range_multiplier
        else:
            sampled_disparities = min_disparity + multiplier * range_multiplier + offset

        return sampled_disparities


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, left_input, right_input, disparity_samples):
        """
        Disparity Sample Cost Evaluator
        Description:
                Given the left image features, right iamge features and the disparity samples, generates:
                    - Warped right image features

        Args:
            :left_input: Left Image Features
            :right_input: Right Image Features
            :disparity_samples:  Disparity Samples

        Returns:
            :warped_right_feature_map: right iamge features warped according to input disparity.
            :left_feature_map: expanded left image features.
        """

        device = left_input.get_device()
        left_y_coordinate = torch.arange(0.0, left_input.size()[3], device=device).repeat(left_input.size()[2])
        left_y_coordinate = left_y_coordinate.view(left_input.size()[2], left_input.size()[3])
        left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=left_input.size()[3] - 1)
        left_y_coordinate = left_y_coordinate.expand(left_input.size()[0], -1, -1)

        right_feature_map = right_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        left_feature_map = left_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])

        disparity_samples = disparity_samples.float()

        right_y_coordinate = left_y_coordinate.expand(
            disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]) - disparity_samples

        right_y_coordinate_1 = right_y_coordinate
        right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=right_input.size()[3] - 1)

        warped_right_feature_map = torch.gather(right_feature_map, dim=4, index=right_y_coordinate.expand(
            right_input.size()[1], -1, -1, -1, -1).permute([1, 0, 2, 3, 4]).long())

        right_y_coordinate_1 = right_y_coordinate_1.unsqueeze(1)
        warped_right_feature_map = (1 - ((right_y_coordinate_1 < 0) +
                                         (right_y_coordinate_1 > right_input.size()[3] - 1)).float()) * \
                                   (warped_right_feature_map) + torch.zeros_like(warped_right_feature_map)

        return warped_right_feature_map, left_feature_map


class Feature_dp(nn.Module):
    def __init__(self, in_plans=32):
        super(Feature_dp, self).__init__()

        self.conv_start_img = nn.Sequential(BasicConv(3, 16, kernel_size=3, padding=1),
                                            BasicConv(16, 16, kernel_size=3, padding=1))

        self.conv_x = nn.Conv2d(in_plans, 16, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=True))

        self.conv_start = nn.Sequential(
            BasicConv(32, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=2, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv0a = BasicConv(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv1a = BasicConv(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(48, 32, deconv=True)
        self.deconv3a = Conv2x(32, 32, deconv=True)
        self.deconv2a = Conv2x(32, 32, deconv=True)
        self.deconv1a = Conv2x(32, 32, deconv=True)

        self.conv1b = Conv2x(32, 32)
        self.conv2b = Conv2x(32, 32)
        self.conv3b = Conv2x(32, 32)
        self.conv4b = Conv2x(32, 48)

        self.deconv4b = Conv2x(48, 32, deconv=True)
        self.deconv3b = Conv2x(32, 32, deconv=True)
        self.deconv2b = Conv2x(32, 32, deconv=True)

        self.deconv1b = Conv2x(32, 32, deconv=True)
        self.deconv0b = Conv2x(32, 32, deconv=True)

        self.conv_out1 = BasicConv(32, 32, kernel_size=3, padding=1)

        self.conv_out2 = BasicConv(64, 32, kernel_size=3, padding=1)

    def forward(self, x, img=None, resolution=1):
        x_ori = x

        if resolution != 1:
            img = F.interpolate(img, [img.shape[-2] // resolution, img.shape[-1] // resolution],
                                mode='bilinear', align_corners=False)

        g = self.conv_start_img(img)

        x1 = self.conv_x(x)
        if g.shape[-1] // x1.shape[-1] != 1:
            factor = g.shape[-1] / x1.shape[-1]
            x1 = F.interpolate(x1, [int(x1.size()[2] * factor), int(x1.size()[3] * factor)], mode='bilinear',
                               align_corners=False)
        x1 = self.bn_relu(x1)

        x = torch.cat([g, x1], dim=1)

        rem_0_0 = self.conv0a(x)

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        #
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)

        #
        rem3 = x
        x = self.conv4b(x, rem4)

        #
        x = self.deconv4b(x, rem3)

        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        x = self.deconv0b(x, rem_0_0)

        x = self.conv_out1(x)
        rem0 = F.interpolate(rem0, [x.shape[2], x.shape[3]], mode='bilinear', align_corners=False)
        x = torch.cat([x, rem0], dim=1)
        x = self.conv_out2(x)

        return x


class DP_Module(nn.Module):
    '''
    expanding sparse-hint
    '''

    def __init__(self, args, feat_channel=32, inner_channel=32):
        super().__init__()
        self.args = args
        self.p = 2 * args.refine_spn_r + 1
        self.patch_size = (self.p, self.p)
        # self.spn_times = args.refine_spn_iter_num

        self.net = Feature_dp(in_plans=feat_channel)

        self.flag_offset = args.refine_spn_offset_flag
        if self.flag_offset:
            self.conv_offset = nn.Sequential(BasicConv(inner_channel, (self.p ** 2), kernel_size=3, padding=1),
                                             BasicConv((self.p) ** 2, (self.p ** 2) * 2, kernel_size=3, padding=1,
                                                       relu=False))
            self.range = args.refine_spn_offset_range

        self.conf_pixel = args.refine_spn_conf_pixel

    def unfold_deformable_x(self, hint=None, feat=None, extra_offset=None):
        N, C_hint, H, W = hint.shape
        _, C_feat, _, _ = feat.shape
        psize = self.patch_size
        ry = psize[0] // 2
        rx = psize[1] // 2
        dilatex, dilatey = 1, 1

        x_grid, y_grid = torch.meshgrid(torch.arange(-rx, rx + 1, dilatex, device=hint.device),
                                        torch.arange(-ry, ry + 1, dilatey, device=hint.device))

        offsets = torch.stack([x_grid, y_grid])
        offsets = offsets.reshape(2, -1).permute(1, 0)
        for d in sorted((0, 2, 3)):
            offsets = offsets.unsqueeze(d)
        offsets = offsets.repeat_interleave(N, dim=0)

        extra_offset = extra_offset.reshape(N, self.p ** 2, 2, H, W).permute(0, 1, 3, 4, 2)  # [N, search_num, 1, 1, 2]
        offsets = offsets + extra_offset

        coords = coords_grid(N, H, W, hint)
        coords = coords.permute(0, 2, 3, 1)  # [N, H, W, 2]
        coords = coords.unsqueeze(1) + offsets
        coords = torch.reshape(coords, (N, -1, W, 2))  # [N, search_num*H, W, 2]

        feat_sample = bilinear_sample_v2(feat, coords, mode='bilinear', padding_mode="border")  # [N,C,num*H,W]
        feat_sample = torch.reshape(feat_sample, (N, C_feat, -1, H, W))
        feat_sample = feat_sample.reshape(N, C_feat, psize[0], psize[1], H, W)
        feat_sample = feat_sample.permute(0, 1, 3, 2, 4, 5).reshape(N, C_feat, -1, H, W)

        hint_sample = bilinear_sample_v2(hint, coords, mode='nearest', padding_mode="border")  # [N,C,num*H,W]
        hint_sample = torch.reshape(hint_sample, (N, -1, H, W))
        hint_sample = hint_sample.reshape(N, psize[0], psize[1], H, W)
        hint_sample = hint_sample.permute(0, 2, 1, 3, 4).reshape(N, -1, H, W)

        return hint_sample, feat_sample

    def forward(self, left_feat, sparse_hint, img=None, resolution=1):
        f_0 = self.net(left_feat, img, resolution=resolution)  # [b,32,h,w]
        b, c, h, w = f_0.shape
        f0_reshape = f_0.view(b, c, -1).permute(0, 2, 1).reshape(b * h * w, 1, c)

        if self.flag_offset:
            offset = self.conv_offset(f_0)
            offset = self.range * (torch.sigmoid(offset) - 0.5) * 2.0  # -1 to 1

            offset[:, self.p ** 2 - 1:self.p ** 2 + 1, :, :] = torch.zeros_like(
                offset[:, self.p ** 2 - 1:self.p ** 2 + 1, :, :])

        feature0_proj = f_0

        if resolution != 1:  # should be >1
            sparse_hint = F.interpolate(sparse_hint,
                                        [sparse_hint.shape[-2] // resolution, sparse_hint.shape[-1] // resolution],
                                        mode='nearest') * (1 / resolution)

        hint_unfold, f_0_unfold = self.unfold_deformable_x(hint=sparse_hint, feat=feature0_proj,
                                                           extra_offset=offset)  # [b,p*p,h,w] [b,32，num,h,w]

        f_0_unfold = f_0_unfold.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, -1)  # [bhw,c,p*p]

        conf = torch.matmul(f0_reshape, f_0_unfold) / (c ** 0.5)  # [b*h*w,1,p*p]

        if self.conf_pixel != 'sparse_valid':
            conf = torch.softmax(conf, dim=-1)

        hint_unfold = hint_unfold.view(b, -1, h * w).permute(0, 2, 1).reshape(b * h * w, -1, 1)  # [b*h*w,p*p,1]
        weight_unfold = (hint_unfold > 0).float()

        if self.conf_pixel == 'sparse_valid':
            conf = torch.where(hint_unfold.permute(0, 2, 1) == 0., torch.zeros_like(conf), conf)
            conf = conf / (torch.sum(conf, dim=-1, keepdim=True) + 1e-7)

        out = torch.matmul(conf, hint_unfold).view(b, h, w, 1).permute(0, 3, 1, 2).contiguous()  # [b,1,h,w]

        weight_out = torch.matmul(conf, weight_unfold).view(b, h, w, 1).permute(0, 3, 1,
                                                                                2).contiguous()  # [b,1,h,w]

        return out, weight_out
