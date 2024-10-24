from abc import abstractmethod

import math
from inspect import isfunction

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import functional as F

from src.networks.diffusion_modules.network_ae import WaveletEncoder
from src.networks.diffusion_modules.dwt import DWTInverse3d
from src.networks.diffusion_modules.sparse_network import SparseComposer
from src.networks.diffusion_modules.fp16_util import (
    convert_module_to_f16,
    convert_module_to_f32,
)
from src.networks.diffusion_modules.gaussian_diffusion import (
    GaussianDiffusion,
    SpacedDiffusion,
    get_named_beta_schedule,
    space_timesteps,
)
from src.networks.diffusion_modules.resample import (
    UniformSampler,
    LossSecondMomentResampler,
)
from src.networks.diffusion_modules.nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
from src.experiments.utils.wavelet_utils import (
    extract_wavelet_coefficients,
    extract_full_indices,
    extract_highs_from_values,
)

# from src.networks.diffusion_modules.point_voxels import PVCNN2
# from src.networks.diffusion_modules.latent_points import LatentArrayTransformer
from einops import rearrange, repeat

# from inspect import isfunction
# from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
# allow_ops_in_compiled_graph()


def expand_dims(tensor, dims):
    for dim in dims:
        tensor = tensor.unsqueeze(dim)
    return tensor


def remove_duplicate(tensor_1, tensor_2):
    # borrow from https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
    combined = torch.cat((tensor_1, tensor_2), dim=0)
    uniques, counts = combined.unique(return_counts=True, dim=0)
    difference = uniques[counts == 1]

    return difference


def pad_with_batch_idx(tensor):
    batch_size = tensor.size(0)
    padding_tensor = (
        torch.arange(batch_size, device=tensor.device)
        .long()
        .unsqueeze(1)
        .repeat(1, tensor.size(1))
        .unsqueeze(2)
    )
    padded_tensor = torch.cat((padding_tensor, tensor), dim=-1)
    return padded_tensor


class get_latent_pt_superres_diffusion_model(nn.Module):
    def __init__(self, args, scale_value=None, avg_value=None):
        super(get_latent_pt_superres_diffusion_model, self).__init__()

        ### Local Model Hyperparameters
        self.avg_value = avg_value
        self.scale_value = scale_value
        self.args = args
        ### sparse models
        self.dwt_sparse_composer = SparseComposer(
            input_shape=[args.resolution, args.resolution, args.resolution],
            J=args.max_depth,
            wave=args.wavelet,
            mode=args.padding_mode,
            inverse_dwt_module=None,
        )

        self.dwt_inverse_3d = DWTInverse3d(
            J=args.max_depth, wave=args.wavelet, mode=args.padding_mode
        )

        ### denoiser
        if self.args.use_point_conv:
            self.denoiser = PVCNN2(
                num_classes=511 + 3,
                embed_dim=args.embed_dim,
                use_att=args.attention,
                context_dim=args.ae_z_dim + 1,
                d_head=args.d_head,
                n_heads=args.n_heads,
                dropout=args.dropout,
                extra_feature_channels=511,
            )
        else:
            self.denoiser = LatentArrayTransformer(
                in_channels=(511 + 3),
                t_channels=self.args.t_channels,
                n_heads=self.args.n_heads,
                d_head=self.args.d_head,
                depth=self.args.depth,
                context_dim=args.ae_z_dim + 1,
            )

        ### encoder
        self.encoder = WaveletEncoder(args=args)

        ## diffusion
        betas = get_named_beta_schedule(
            self.args.diffusion_beta_schedule,
            self.args.diffusion_step,
            self.args.diffusion_scale_ratio,
        )
        self.diffusion_module = GaussianDiffusion(
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
            rescale_timesteps=(
                self.args.diffusion_rescale_timestep
                if hasattr(self.args, "diffusion_rescale_timestep")
                else False
            ),
        )

        self.inference_diffusion_module = SpacedDiffusion(
            use_timesteps=space_timesteps(
                self.args.diffusion_step, [self.args.diffusion_rescale_timestep]
            ),
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
        )

        ## sample
        if self.args.diffusion_sampler == "uniform":
            self.sampler = UniformSampler(self.diffusion_module)
        elif self.args.diffusion_sampler == "second-order":
            self.sampler = LossSecondMomentResampler(self.diffusion_module)
        else:
            raise Exception("Unknown Sampler.....")

        ### progressive
        if self.args.use_progressive_train:
            if (
                hasattr(self.args, "start_from_coordinate")
                and self.args.start_from_coordinate
            ):
                self.current_stage = -1
            else:
                self.current_stage = 0

    def training_losses(self, low, high_indices, high_values):

        ### spatial resolution
        spatial_resolution = self.dwt_sparse_composer.shape_list[-1][0]

        ### latent code
        context = self.encoder(low)

        ### compute coordindates
        high_indices_xyz = (
            high_indices[:, :, -1, 1:].float() / spatial_resolution
        ) - 0.5

        ###
        if self.args.use_shift_mean:
            high_values = high_values - self.avg_value.unsqueeze(0).unsqueeze(0).to(
                high_values.device
            )

        if self.args.use_normalization:
            high_values = high_values / self.scale_value.to(high_values.device)

        latent_pts = torch.cat((high_indices_xyz, high_values), dim=2)

        ### sample time
        # loss computation
        batch_size = low.size(0)
        t, weights = self.sampler.sample(batch_size, device=low.device)

        model_kawrgs = {"context": context}

        if self.args.use_progressive_train:
            model_kawrgs["current_stage"] = self.current_stage

        if self.args.use_balance_loss:
            model_kawrgs["use_balance_loss"] = True

        loss = self.diffusion_module.training_losses(
            self.denoiser, x_start=latent_pts, t=t, model_kwargs=model_kawrgs
        )

        loss = torch.mean(weights * loss["loss"])

        return loss

    def inference(self, low):

        ### assume batch = 1
        ### spatial resolution
        spatial_resolution = self.dwt_sparse_composer.shape_list[-1][0]

        batch_size = low.size(0)
        context = self.encoder(low)
        model_kawrgs = {"context": context}
        generation_shapes = (batch_size, self.args.point_num, 511 + 3)
        pred, _ = self.inference_diffusion_module.p_sample_loop(
            model=self.denoiser,
            shape=generation_shapes,
            device=low.device,
            clip_denoised=False,
            progress=True,
            model_kwargs=model_kawrgs,
        )

        indices, values = pred[:, :, :3], pred[:, :, 3:]  ##

        if self.args.use_progressive_train:
            order = (2 ** (self.current_stage + 1)) ** 3
            idx = order - 1
            values[:, :, :-idx] = 0

        if self.args.use_normalization:
            values = values * self.scale_value

        if self.args.use_shift_mean:
            values = values + self.avg_value.unsqueeze(0).unsqueeze(0)

        indices = torch.clip(
            torch.floor((indices + 0.5) * spatial_resolution), 0, spatial_resolution - 1
        ).long()

        highs_full = torch.zeros(
            (
                batch_size,
                self.dwt_sparse_composer.shape_list[-1][0],
                self.dwt_sparse_composer.shape_list[-1][1],
                self.dwt_sparse_composer.shape_list[-1][2],
                511,
            )
        ).to(
            low.device
        )  # hard-code test
        batch_pad = (
            torch.arange(batch_size, device=indices.device)
            .unsqueeze(1)
            .repeat((1, indices.size(1)))
            .unsqueeze(2)
            .long()
        )  # B * P * 1
        high_indices_filled = torch.cat((batch_pad, indices), dim=2)  # B * P * 4
        high_indices_filled = high_indices_filled.reshape((-1, 4))

        highs_full[
            high_indices_filled[:, 0],
            high_indices_filled[:, 1],
            high_indices_filled[:, 2],
            high_indices_filled[:, 3],
            :,
        ] = values.reshape((-1, 511))

        ## high full is ready

        highs_full = highs_full.reshape((-1, 511))  # reshape
        highs_full_indices = extract_full_indices(
            device=highs_full.device,
            max_depth=self.args.max_depth,
            shape_list=self.dwt_sparse_composer.shape_list,
        )

        highs_full_recon = extract_highs_from_values(
            highs_full_indices,
            highs_full,
            self.args.max_depth,
            self.dwt_sparse_composer.shape_list,
        )

        return highs_full_recon


class get_latent_superres_diffusion_model(nn.Module):
    def __init__(self, args, scale_value=None, avg_value=None):
        super(get_latent_superres_diffusion_model, self).__init__()

        ### Local Model Hyperparameters
        self.args = args

        ### sparse models
        self.dwt_sparse_composer = SparseComposer(
            input_shape=[args.resolution, args.resolution, args.resolution],
            J=args.max_depth,
            wave=args.wavelet,
            mode=args.padding_mode,
            inverse_dwt_module=None,
        )

        self.dwt_inverse_3d = DWTInverse3d(
            J=args.max_depth, wave=args.wavelet, mode=args.padding_mode
        )

        self.unet = UNetModel(
            in_channels=512,
            model_channels=self.args.unet_model_channels,
            out_channels=511,
            num_res_blocks=self.args.unet_num_res_blocks,
            channel_mult=self.args.unet_channel_mult,
            attention_resolutions=self.args.attention_resolutions,
            dropout=0,
            dims=3,
        )

        ## diffusion
        betas = get_named_beta_schedule(
            self.args.diffusion_beta_schedule,
            self.args.diffusion_step,
            self.args.diffusion_scale_ratio,
        )
        self.diffusion_module = GaussianDiffusion(
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
            rescale_timesteps=(
                self.args.diffusion_rescale_timestep
                if hasattr(self.args, "diffusion_rescale_timestep")
                else False
            ),
        )

        self.inference_diffusion_module = SpacedDiffusion(
            use_timesteps=space_timesteps(
                self.args.diffusion_step, [self.args.diffusion_rescale_timestep]
            ),
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
        )

        ## sample
        if self.args.diffusion_sampler == "uniform":
            self.sampler = UniformSampler(self.diffusion_module)
        elif self.args.diffusion_sampler == "second-order":
            self.sampler = LossSecondMomentResampler(self.diffusion_module)
        else:
            raise Exception("Unknown Sampler.....")

        if self.args.use_progressive_train:
            self.current_stage = self.args.start_stage

        self.avg_value = avg_value
        self.scale_value = scale_value

    def training_losses(self, low, high_indices, high_values):

        # loss computation
        batch_size = low.size(0)
        t, weights = self.sampler.sample(batch_size, device=low.device)

        if self.args.use_shift_mean:
            high_values = high_values - self.avg_value.unsqueeze(0).unsqueeze(0).to(
                high_values.device
            )

        highs_full = torch.zeros(
            (
                batch_size,
                self.dwt_sparse_composer.shape_list[-1][0],
                self.dwt_sparse_composer.shape_list[-1][1],
                self.dwt_sparse_composer.shape_list[-1][2],
                511,
            )
        ).to(
            low.device
        )  # hard-code test
        high_indices_filled = high_indices[:, :, -1, 1:].long()
        batch_pad = (
            torch.arange(batch_size, device=high_indices_filled.device)
            .unsqueeze(1)
            .repeat((1, high_indices_filled.size(1)))
            .unsqueeze(2)
            .long()
        )  # B * P * 1
        high_indices_filled = torch.cat(
            (batch_pad, high_indices_filled), dim=2
        )  # B * P * 4
        high_indices_filled = high_indices_filled.reshape((-1, 4))
        highs_full[
            high_indices_filled[:, 0],
            high_indices_filled[:, 1],
            high_indices_filled[:, 2],
            high_indices_filled[:, 3],
            :,
        ] = high_values.reshape((-1, 511))

        highs_full = torch.permute(highs_full, (0, 4, 1, 2, 3))

        model_kawrgs = {"low": low}

        if self.args.use_progressive_train:
            model_kawrgs["current_stage"] = self.current_stage

        if self.args.use_sample_training:
            non_empty_indices = high_indices[:, :, -1, 1:].long()

            ### get indices
            highs_full_indices = extract_full_indices(
                device=non_empty_indices.device,
                max_depth=self.args.max_depth,
                shape_list=self.dwt_sparse_composer.shape_list,
            )
            highs_full_indices = highs_full_indices[:, -1, 1:].long()

            if self.args.use_sample_threshold:
                non_empty_indices_filtered = []
                for idx in range(batch_size):
                    high_values_last = high_values[
                        idx, :, -7:
                    ]  ### get last 7 dimensions
                    high_values_max, _ = torch.max(torch.abs(high_values_last), dim=0)
                    high_values_keep = (
                        torch.abs(high_values_last)
                        > high_values_max.unsqueeze(0)
                        * self.args.sample_threshold_ratio
                    )  ## keep those
                    high_values_keep = (
                        torch.max(high_values_keep, dim=1)[0] > 0
                    )  ## keep indices
                    non_empty_index = non_empty_indices[idx][high_values_keep]
                    non_empty_indices_filtered.append(non_empty_index)

                non_empty_indices = non_empty_indices_filtered

            ### only can do it with a for loop right now due to unknown order of unique
            empty_indices = []
            for idx in range(batch_size):
                non_empty_index = non_empty_indices[idx]
                empty_indices_idx = remove_duplicate(
                    highs_full_indices, non_empty_index
                )
                indices_perm = torch.randperm(empty_indices_idx.size(0))
                empty_indices_idx = empty_indices_idx[indices_perm]
                empty_indices_idx = empty_indices_idx[
                    : non_empty_index.size(0)
                ].unsqueeze(0)
                empty_indices.append(empty_indices_idx)

            ## padding
            if self.args.use_sample_threshold:
                non_empty_indices = self.pad_with_batch_idx_list(non_empty_indices)
                empty_indices = self.pad_with_batch_idx_list(
                    [empty_index.squeeze(0) for empty_index in empty_indices]
                )
                training_indices = [
                    torch.cat((non_empty_indices[i], empty_indices[i]), dim=0)
                    for i in range(len(non_empty_indices))
                ]
            else:
                non_empty_indices = pad_with_batch_idx(non_empty_indices)
                empty_indices = torch.cat(empty_indices, dim=0)
                empty_indices = pad_with_batch_idx(empty_indices)
                training_indices = torch.cat((empty_indices, non_empty_indices), dim=1)

            model_kawrgs["training_indices"] = training_indices

        loss = self.diffusion_module.training_losses(
            self.unet, x_start=highs_full, t=t, model_kwargs=model_kawrgs
        )
        loss = torch.mean(weights * loss["loss"])

        return loss

    def pad_with_batch_idx_list(self, non_empty_indices):
        non_empty_indices_padding = [
            torch.full(
                (non_empty_index.size(0), 1),
                idx,
                device=non_empty_indices[idx].device,
                dtype=torch.long,
            )
            for idx, non_empty_index in enumerate(non_empty_indices)
        ]
        non_empty_indices = [
            torch.cat((non_empty_indices_padding[idx], non_empty_indices[idx]), dim=1)
            for idx in range(len(non_empty_indices))
        ]
        return non_empty_indices

    def inference(self, low):

        ## compute inference
        shape = (low.size(0), 511, low.size(2), low.size(3), low.size(4))
        model_kawrgs = {"low": low}
        pred, _ = self.inference_diffusion_module.p_sample_loop(
            model=self.unet,
            shape=shape,
            device=low.device,
            clip_denoised=False,
            progress=True,
            model_kwargs=model_kawrgs,
        )

        if self.args.use_shift_mean:
            pred = pred + self.avg_value.unsqueeze(0).unsqueeze(2).unsqueeze(
                3
            ).unsqueeze(4).to(pred.device)

        if self.args.use_progressive_train:
            order = (2 ** (self.current_stage + 1)) ** 3
            idx = order - 1
            pred[:, :-idx, :, :, :] = 0

        ## assume pred batch size = 1
        pred = torch.permute(pred, (0, 2, 3, 4, 1))  ## re-organize the dimension

        highs_full = pred.reshape((-1, 511))  # reshape

        highs_full_indices = extract_full_indices(
            device=highs_full.device,
            max_depth=self.args.max_depth,
            shape_list=self.dwt_sparse_composer.shape_list,
        )

        highs_full_recon = extract_highs_from_values(
            highs_full_indices,
            highs_full,
            self.args.max_depth,
            self.dwt_sparse_composer.shape_list,
        )

        return highs_full_recon


class get_latent_progressive_diffusion_model(nn.Module):
    def __init__(self, args):
        super(get_latent_progressive_diffusion_model, self).__init__()

        ### Local Model Hyperparameters
        self.args = args

        ### sparse models
        self.dwt_sparse_composer = SparseComposer(
            input_shape=[args.resolution, args.resolution, args.resolution],
            J=args.max_depth,
            wave=args.wavelet,
            mode=args.padding_mode,
            inverse_dwt_module=None,
        )

        self.dwt_inverse_3d = DWTInverse3d(
            J=args.max_depth, wave=args.wavelet, mode=args.padding_mode
        )

        self.unet = UNetModel(
            in_channels=512,
            model_channels=self.args.unet_model_channels,
            out_channels=512,
            num_res_blocks=self.args.unet_num_res_blocks,
            channel_mult=self.args.unet_channel_mult,
            attention_resolutions=self.args.attention_resolutions,
            dropout=0,
            dims=3,
        )

        ## diffusion
        betas = get_named_beta_schedule(
            self.args.diffusion_beta_schedule,
            self.args.diffusion_step,
            self.args.diffusion_scale_ratio,
        )
        self.diffusion_module = GaussianDiffusion(
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
            rescale_timesteps=(
                self.args.diffusion_rescale_timestep
                if hasattr(self.args, "diffusion_rescale_timestep")
                else False
            ),
        )

        self.inference_diffusion_module = SpacedDiffusion(
            use_timesteps=space_timesteps(
                self.args.diffusion_step, [self.args.diffusion_rescale_timestep]
            ),
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
        )

        ## sample
        if self.args.diffusion_sampler == "uniform":
            self.sampler = UniformSampler(self.diffusion_module)
        elif self.args.diffusion_sampler == "second-order":
            self.sampler = LossSecondMomentResampler(self.diffusion_module)
        else:
            raise Exception("Unknown Sampler.....")

        self.avg_value = torch.from_numpy(
            np.array([2.20] + [0] * 511)
        )  ### HARD_CODE first
        self.scale_value = torch.ones_like(self.avg_value)

    def training_losses(
        self,
        low,
        high_indices,
        high_values,
        current_stage,
        high_indices_empty=None,
        high_values_mask=None,
    ):

        # loss computation
        batch_size = low.size(0)
        t, weights = self.sampler.sample(batch_size, device=low.device)

        highs_full = torch.zeros(
            (
                batch_size,
                self.dwt_sparse_composer.shape_list[-1][0],
                self.dwt_sparse_composer.shape_list[-1][1],
                self.dwt_sparse_composer.shape_list[-1][2],
                511,
            )
        ).to(
            low.device
        )  # hard-code test

        if hasattr(self.args, "sanity_test") and self.args.sanity_test:
            print("Sanity test Filled by zero")
        else:
            high_indices_filled = high_indices[:, :, -1, 1:].long()
            batch_pad = (
                torch.arange(batch_size, device=high_indices_filled.device)
                .unsqueeze(1)
                .repeat((1, high_indices_filled.size(1)))
                .unsqueeze(2)
                .long()
            )  # B * P * 1
            high_indices_filled = torch.cat(
                (batch_pad, high_indices_filled), dim=2
            )  # B * P * 4
            high_indices_filled = high_indices_filled.reshape((-1, 4))
            highs_full[
                high_indices_filled[:, 0],
                high_indices_filled[:, 1],
                high_indices_filled[:, 2],
                high_indices_filled[:, 3],
                :,
            ] = high_values.reshape((-1, 511))

        highs_full = torch.permute(highs_full, (0, 4, 1, 2, 3))
        data_samples = torch.cat((low, highs_full), dim=1)

        if self.args.use_shift_mean:
            avg_value = (
                self.avg_value.unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .to(data_samples.device)
            )
            avg_value = avg_value.type(data_samples.dtype)
            data_samples = data_samples - avg_value

        model_kawrgs = {"progressive_generator": True}

        model_kawrgs["current_stage"] = current_stage

        if self.args.use_sample_training:

            if (
                hasattr(self.args, "use_batched_threshold")
                and self.args.use_batched_threshold
            ):
                assert high_indices_empty is not None and high_values_mask is not None

                ## compute training indices
                non_empty_indices = high_indices[:, :, -1, 1:].long()
                training_indices = torch.cat(
                    (non_empty_indices, high_indices_empty), dim=1
                )
                training_indices = pad_with_batch_idx(training_indices)

                model_kawrgs["training_indices"] = training_indices

                high_values_mask = high_values_mask.repeat(1, 2)
                model_kawrgs["high_values_mask"] = high_values_mask
            else:
                non_empty_indices = high_indices[:, :, -1, 1:].long()

                ### get indices
                highs_full_indices = extract_full_indices(
                    device=non_empty_indices.device,
                    max_depth=self.args.max_depth,
                    shape_list=self.dwt_sparse_composer.shape_list,
                )
                highs_full_indices = highs_full_indices[:, -1, 1:].long()

                if self.args.use_sample_threshold:
                    non_empty_indices_filtered = []
                    for idx in range(batch_size):
                        high_values_last = high_values[
                            idx, :, -7:
                        ]  ### get last 7 dimensions
                        high_values_max, _ = torch.max(
                            torch.abs(high_values_last), dim=0
                        )
                        high_values_keep = (
                            torch.abs(high_values_last)
                            > high_values_max.unsqueeze(0)
                            * self.args.sample_threshold_ratio
                        )  ## keep those
                        high_values_keep = (
                            torch.max(high_values_keep, dim=1)[0] > 0
                        )  ## keep indices
                        non_empty_index = non_empty_indices[idx][high_values_keep]
                        non_empty_indices_filtered.append(non_empty_index)

                    non_empty_indices = non_empty_indices_filtered

                ### only can do it with a for loop right now due to unknown order of unique
                empty_indices = []
                for idx in range(batch_size):
                    non_empty_index = non_empty_indices[idx]
                    empty_indices_idx = remove_duplicate(
                        highs_full_indices, non_empty_index
                    )
                    indices_perm = torch.randperm(empty_indices_idx.size(0))
                    empty_indices_idx = empty_indices_idx[indices_perm]
                    empty_indices_idx = empty_indices_idx[
                        : non_empty_index.size(0)
                    ].unsqueeze(0)
                    empty_indices.append(empty_indices_idx)

                ## padding
                if self.args.use_sample_threshold:
                    non_empty_indices = self.pad_with_batch_idx_list(non_empty_indices)
                    empty_indices = self.pad_with_batch_idx_list(
                        [empty_index.squeeze(0) for empty_index in empty_indices]
                    )
                    training_indices = [
                        torch.cat((non_empty_indices[i], empty_indices[i]), dim=0)
                        for i in range(len(non_empty_indices))
                    ]
                else:
                    non_empty_indices = pad_with_batch_idx(non_empty_indices)
                    empty_indices = torch.cat(empty_indices, dim=0)
                    empty_indices = pad_with_batch_idx(empty_indices)
                    training_indices = torch.cat(
                        (empty_indices, non_empty_indices), dim=1
                    )

                model_kawrgs["training_indices"] = training_indices

        model_kawrgs["no_rebalance_loss"] = self.args.no_rebalance_loss

        loss = self.diffusion_module.training_losses(
            self.unet, x_start=data_samples, t=t, model_kwargs=model_kawrgs
        )
        loss["loss"] = torch.mean(weights * loss["loss"])
        loss["base_loss"] = torch.mean(weights * loss["base_loss"])
        for idx in range(current_stage):
            loss[f"loss_{idx+1}"] = torch.mean(weights * loss[f"loss_{idx+1}"])

        return loss

    def pad_with_batch_idx_list(self, non_empty_indices):
        non_empty_indices_padding = [
            torch.full(
                (non_empty_index.size(0), 1),
                idx,
                device=non_empty_indices[idx].device,
                dtype=torch.long,
            )
            for idx, non_empty_index in enumerate(non_empty_indices)
        ]
        non_empty_indices = [
            torch.cat((non_empty_indices_padding[idx], non_empty_indices[idx]), dim=1)
            for idx in range(len(non_empty_indices))
        ]
        return non_empty_indices

    def inference(self, low, local_rank, current_stage):

        ## compute inference
        shape = (low.size(0), 512, low.size(2), low.size(3), low.size(4))
        model_kawrgs = {}
        pred, _ = self.inference_diffusion_module.p_sample_loop(
            model=self.unet,
            shape=shape,
            device=low.device,
            clip_denoised=False,
            progress=True,
            model_kwargs=model_kawrgs,
        )

        if self.args.use_shift_mean:
            avg_value = (
                self.avg_value.unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .to(pred.device)
            )
            avg_value = avg_value.type(pred.dtype)
            pred = pred + avg_value

        order = (2**current_stage) ** 3
        idx = order - 1
        pred[:, 1 : pred.size(1) - idx, :, :, :] = 0  # filter other stuff

        ## assume pred batch size = 1 by filtering local rank
        low_pred, pred = (
            pred[local_rank : local_rank + 1, :1],
            pred[local_rank : local_rank + 1, 1:],
        )
        pred = torch.permute(pred, (0, 2, 3, 4, 1))  ## re-organize the dimension

        highs_full = pred.reshape((-1, 511))  # reshape

        highs_full_indices = extract_full_indices(
            device=highs_full.device,
            max_depth=self.args.max_depth,
            shape_list=self.dwt_sparse_composer.shape_list,
        )

        highs_full_recon = extract_highs_from_values(
            highs_full_indices,
            highs_full,
            self.args.max_depth,
            self.dwt_sparse_composer.shape_list,
        )

        return low_pred, highs_full_recon


class get_latent_superres_model(nn.Module):
    def __init__(self, args):
        super(get_latent_superres_model, self).__init__()

        ### Local Model Hyperparameters
        self.args = args

        ### sparse models
        self.dwt_sparse_composer = SparseComposer(
            input_shape=[args.resolution, args.resolution, args.resolution],
            J=args.max_depth,
            wave=args.wavelet,
            mode=args.padding_mode,
            inverse_dwt_module=None,
        )

        self.dwt_inverse_3d = DWTInverse3d(
            J=args.max_depth, wave=args.wavelet, mode=args.padding_mode
        )

        self.unet = MyUNetModel(
            in_channels=1,
            spatial_size=self.dwt_sparse_composer.shape_list[-1][0],
            model_channels=self.args.unet_model_channels,
            out_channels=511,
            num_res_blocks=self.args.unet_num_res_blocks,
            channel_mult=self.args.unet_channel_mult,
            attention_resolutions=self.args.attention_resolutions,
            dropout=0,
            dims=3,
        )

    def inference(self, low):
        pred = self.unet(low)
        ## assume pred batch size = 1
        pred = torch.permute(pred, (0, 2, 3, 4, 1))  ## re-organize the dimension

        highs_full = pred.reshape((-1, 511))  # reshape
        highs_full_indices = extract_full_indices(
            device=highs_full.device,
            max_depth=self.args.max_depth,
            shape_list=self.dwt_sparse_composer.shape_list,
        )

        highs_full_recon = extract_highs_from_values(
            highs_full_indices,
            highs_full,
            self.args.max_depth,
            self.dwt_sparse_composer.shape_list,
        )

        return highs_full_recon

    def training_losses(self, low, high_indices, high_values):

        batch_size = low.size(0)
        pred = self.unet(low)

        if self.args.use_sample_training:
            non_empty_indices = high_indices[:, :, -1, 1:].long()

            ### get indices
            highs_full_indices = extract_full_indices(
                device=non_empty_indices.device,
                max_depth=self.args.max_depth,
                shape_list=self.dwt_sparse_composer.shape_list,
            )
            highs_full_indices = highs_full_indices[:, -1, 1:].long()

            if self.args.use_sample_threshold:
                non_empty_indices_filtered = []
                non_empty_high_values_filtered = []
                for idx in range(batch_size):
                    high_values_last = high_values[
                        idx, :, -7:
                    ]  ### get last 7 dimensions
                    high_values_max, _ = torch.max(torch.abs(high_values_last), dim=0)
                    high_values_keep = (
                        torch.abs(high_values_last)
                        > high_values_max.unsqueeze(0)
                        * self.args.sample_threshold_ratio
                    )  ## keep those
                    high_values_keep = (
                        torch.max(high_values_keep, dim=1)[0] > 0
                    )  ## keep indices
                    non_empty_index = non_empty_indices[idx][high_values_keep]
                    non_empty_indices_filtered.append(non_empty_index)

                    ### target values
                    non_empty_high_value = high_values[idx, high_values_keep]
                    non_empty_high_values_filtered.append(non_empty_high_value)

                non_empty_high_values_filtered = torch.cat(
                    non_empty_high_values_filtered, dim=0
                )
                non_empty_indices = non_empty_indices_filtered

            ### only can do it with a for loop right now due to unknown order of unique
            empty_indices = []
            for idx in range(batch_size):
                non_empty_index = non_empty_indices[idx]
                empty_indices_idx = remove_duplicate(
                    highs_full_indices, non_empty_index
                )
                indices_perm = torch.randperm(empty_indices_idx.size(0))
                empty_indices_idx = empty_indices_idx[indices_perm]
                empty_indices_idx = empty_indices_idx[
                    : non_empty_index.size(0)
                ].unsqueeze(0)
                empty_indices.append(empty_indices_idx)

            ## padding
            if self.args.use_sample_threshold:
                non_empty_indices = self.pad_with_batch_idx_list(
                    non_empty_indices
                ).long()
                empty_indices = self.pad_with_batch_idx_list(
                    [empty_index.squeeze(0) for empty_index in empty_indices]
                ).long()
            else:
                non_empty_indices = pad_with_batch_idx(non_empty_indices).reshape(
                    (-1, 4)
                )
                empty_indices = torch.cat(empty_indices, dim=0)
                empty_indices = pad_with_batch_idx(empty_indices).reshape((-1, 4))

            pred = torch.permute(pred, (0, 2, 3, 4, 1))
            non_empty_values = pred[
                non_empty_indices[:, 0],
                non_empty_indices[:, 1],
                non_empty_indices[:, 2],
                non_empty_indices[:, 3],
            ]
            empty_values = pred[
                empty_indices[:, 0],
                empty_indices[:, 1],
                empty_indices[:, 2],
                empty_indices[:, 3],
            ]
            pred_values = torch.cat((non_empty_values, empty_values), dim=0)

            if self.args.use_sample_threshold:
                high_values = non_empty_high_values_filtered
            else:
                high_values = high_values.reshape((-1, 511))

            target_values = torch.cat(
                (high_values, torch.zeros_like(high_values, device=high_values.device)),
                dim=0,
            )
            mse_loss = torch.nn.functional.mse_loss(pred_values, target_values)
            return mse_loss

        else:
            highs_full = torch.zeros(
                (
                    batch_size,
                    self.dwt_sparse_composer.shape_list[-1][0],
                    self.dwt_sparse_composer.shape_list[-1][1],
                    self.dwt_sparse_composer.shape_list[-1][2],
                    511,
                )
            ).to(
                low.device
            )  # hard-code test
            high_indices = high_indices[:, :, -1, 1:].long()
            batch_pad = (
                torch.arange(batch_size, device=high_indices.device)
                .unsqueeze(1)
                .repeat((1, high_indices.size(1)))
                .unsqueeze(2)
                .long()
            )  # B * P * 1
            high_indices = torch.cat((batch_pad, high_indices), dim=2)  # B * P * 4
            high_indices = high_indices.reshape((-1, 4))
            highs_full[
                high_indices[:, 0],
                high_indices[:, 1],
                high_indices[:, 2],
                high_indices[:, 3],
                :,
            ] = high_values.reshape((-1, 511))

            pred = torch.permute(pred, (0, 2, 3, 4, 1))
            mse_loss = torch.nn.functional.mse_loss(pred, highs_full)
            return mse_loss

    def pad_with_batch_idx_list(self, non_empty_indices):
        non_empty_indices_padding = [
            torch.full(
                (non_empty_index.size(0), 1),
                idx,
                device=non_empty_indices[idx].device,
                dtype=torch.long,
            )
            for idx, non_empty_index in enumerate(non_empty_indices)
        ]
        non_empty_indices = [
            torch.cat((non_empty_indices_padding[idx], non_empty_indices[idx]), dim=1)
            for idx in range(len(non_empty_indices))
        ]
        non_empty_indices = torch.cat(non_empty_indices, dim=0)
        return non_empty_indices


class get_superres_model(nn.Module):
    def __init__(self, args):
        super(get_superres_model, self).__init__()

        ### Local Model Hyperparameters
        self.args = args

        ### sparse models
        self.dwt_sparse_composer = SparseComposer(
            input_shape=[args.resolution, args.resolution, args.resolution],
            J=args.max_depth,
            wave=args.wavelet,
            mode=args.padding_mode,
            inverse_dwt_module=None,
        )

        self.dwt_inverse_3d = DWTInverse3d(
            J=args.max_depth, wave=args.wavelet, mode=args.padding_mode
        )

        self.unet = MyUNetModel(
            in_channels=1,
            spatial_size=self.dwt_sparse_composer.shape_list[-1][0],
            model_channels=self.args.unet_model_channels,
            out_channels=7,
            num_res_blocks=self.args.unet_num_res_blocks,
            channel_mult=self.args.unet_channel_mult,
            attention_resolutions=self.args.attention_resolutions,
            dropout=0,
            dims=3,
        )

    def inference(self, data_input):
        pred = self.unet(data_input)
        return pred

    def training_losses(self, data_input, data_target):
        pred = self.unet(data_input)
        mse_loss = torch.nn.functional.mse_loss(pred, data_target)
        return mse_loss


class get_model(nn.Module):
    def __init__(self, args, scale_value, avg_value):
        super(get_model, self).__init__()

        ### Local Model Hyperparameters
        self.args = args

        ### sparse models
        self.dwt_sparse_composer = SparseComposer(
            input_shape=[args.resolution, args.resolution, args.resolution],
            J=args.max_depth,
            wave=args.wavelet,
            mode=args.padding_mode,
            inverse_dwt_module=None,
        )
        self.dwt_inverse_3d = DWTInverse3d(
            J=args.max_depth, wave=args.wavelet, mode=args.padding_mode
        )

        ### the networks
        self.unet = UNetModel(
            in_channels=1 + 7 * (self.args.max_depth - self.args.keep_level),
            model_channels=self.args.unet_model_channels,
            out_channels=1 + 7 * (self.args.max_depth - self.args.keep_level),
            num_res_blocks=self.args.unet_num_res_blocks,
            channel_mult=self.args.unet_channel_mult_low,
            attention_resolutions=self.args.attention_resolutions,
            use_scale_shift_norm=True,
            dropout=0,
            dims=3,
            activation=None,
        )

        ## diffusion
        betas = get_named_beta_schedule(
            self.args.diffusion_beta_schedule,
            self.args.diffusion_step,
            self.args.diffusion_scale_ratio,
        )
        self.diffusion_module = GaussianDiffusion(
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
            rescale_timesteps=(
                self.args.diffusion_rescale_timestep
                if hasattr(self.args, "diffusion_rescale_timestep")
                else False
            ),
        )

        self.inference_diffusion_module = SpacedDiffusion(
            use_timesteps=space_timesteps(
                self.args.diffusion_step, [self.args.diffusion_rescale_timestep]
            ),
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
        )

        ## sample
        if self.args.diffusion_sampler == "uniform":
            self.sampler = UniformSampler(self.diffusion_module)
        elif self.args.diffusion_sampler == "second-order":
            self.sampler = LossSecondMomentResampler(self.diffusion_module)
        else:
            raise Exception("Unknown Sampler.....")

        ## min max
        if self.args.use_normalization:
            self.scale_value = (
                scale_value.unsqueeze(2).unsqueeze(3).unsqueeze(4).float()
            )

        ## mean
        if args.use_shift_mean:
            self.avg_value = avg_value.unsqueeze(2).unsqueeze(3).unsqueeze(4).float()

    # def forward(self, data_input):
    #     latent_codes = self.encoder(data_input)
    #
    #     return dec, diff, info

    def inference(self, data_input):
        low, highs = self.generate_coefficients(data_input)

        sdf_recon = self.dwt_inverse_3d((low, highs))

        return sdf_recon.detach().cpu().numpy()

    def generate_coefficients(self, data_input):
        ## compute inference
        samples, _ = self.inference_diffusion_module.p_sample_loop(
            model=self.unet,
            shape=data_input.size(),
            device=data_input.device,
            clip_denoised=False,
            progress=True,
        )
        ## unormalize
        if self.args.use_normalization:
            samples = samples * self.scale_value.to(samples.device)
        ## transform back
        if self.args.use_shift_mean:
            samples = samples + self.avg_value.to(samples.device)
        low, highs = extract_wavelet_coefficients(
            samples.detach().cpu().numpy(),
            spatial_shapes=self.dwt_sparse_composer.shape_list,
            max_depth=self.args.max_depth,
            keep_level=self.args.keep_level,
            device=torch.device("cuda"),
        )
        return low, highs

    def training_losses(self, data_input):

        # loss computation
        t, weights = self.sampler.sample(data_input.size(0), device=data_input.device)

        if self.args.use_shift_mean:
            data_input = data_input - self.avg_value.to(data_input.device)

        if self.args.use_normalization:
            data_input = data_input / self.scale_value.to(data_input.device)

        loss = self.diffusion_module.training_losses(self.unet, x_start=data_input, t=t)
        loss = torch.mean(weights * loss["loss"])

        return loss


### code from latent shape


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimeConstepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, z_sem):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimeConstepEmbedSequential(nn.Sequential, TimeConstepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, z_sem):
        for layer in self:
            if isinstance(layer, TimeConstepBlock):
                x = layer(x, emb, z_sem)
            else:
                x = layer(x)
        return x


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        activation=SiLU(),
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            activation,
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            activation,
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            activation,
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Con_ResBlock(TimeConstepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        ae_latent_dim=256,
        activation=SiLU(),
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            activation,
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            activation,
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            activation,
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        self.linear_sem = linear(ae_latent_dim, self.out_channels)

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, latent_codes):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward,
            (x, emb, latent_codes),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, x, emb, z_sem):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            z_sem = torch.nn.functional.adaptive_avg_pool3d(z_sem, (1, 1, 1))[
                :, :, 0, 0, 0
            ]
            z_s = self.linear_sem(z_sem)
            while len(z_s.shape) < len(scale.shape):
                z_s = z_s[..., None]
            h = (out_norm(h) * (1 + scale) + shift) * z_s
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.
        Meant to be used like:
            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )
        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial**2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        activation=None,
    ):
        super().__init__()

        self.activation = activation if activation is not None else SiLU()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            self.activation,
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        activation=self.activation,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                activation=self.activation,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                activation=self.activation,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        activation=self.activation,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            self.activation,
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return torch.float32  # FIXED
        # return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None, low_cond=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param low_cond: an [N x C x ...]  Tensor of condition.
        :return: an [N x C x ...] Tensor of outputs.
        """

        ## concat the condition
        if low_cond is not None:
            x = th.cat((x, low_cond), dim=1)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:

            # handling for non-even inputs
            # h = F.interpolate(h, size= hs[-1].size()[-3:], mode='trilinear')
            if hs[-1].size(-1) < h.size(-1):
                h = h[..., :-1]
            if hs[-1].size(-2) < h.size(-2):
                h = h[..., :-1, :]
            if hs[-1].size(-3) < h.size(-3):
                h = h[..., :-1, :, :]

            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class Con_UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        activation=None,
    ):
        super().__init__()

        self.activation = activation if activation is not None else SiLU()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            self.activation,
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimeConstepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    Con_ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        activation=self.activation,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimeConstepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimeConstepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimeConstepEmbedSequential(
            Con_ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                activation=self.activation,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            Con_ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                activation=self.activation,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    Con_ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        activation=self.activation,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimeConstepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            self.activation,
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return torch.float32  # FIXED
        # return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None, low_cond=None, latent_codes=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param low_cond: an [N x C x ...]  Tensor of condition.
        :return: an [N x C x ...] Tensor of outputs.
        """

        ## concat the condition
        if low_cond is not None:
            x = th.cat((x, low_cond), dim=1)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb, latent_codes)
            hs.append(h)
        h = self.middle_block(h, emb, latent_codes)
        for module in self.output_blocks:

            # handling for non-even inputs
            # h = F.interpolate(h, size= hs[-1].size()[-3:], mode='trilinear')
            if hs[-1].size(-1) < h.size(-1):
                h = h[..., :-1]
            if hs[-1].size(-2) < h.size(-2):
                h = h[..., :-1, :]
            if hs[-1].size(-3) < h.size(-3):
                h = h[..., :-1, :, :]

            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb, latent_codes)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class MyResBlock(torch.nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        sp,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        activation=SiLU(),
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=[sp, sp, sp]),
            activation,
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.out_layers = nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=[sp, sp, sp]),
            activation,
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h

    def _forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class MyUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        spatial_size,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        activation=None,
    ):
        super().__init__()

        self.activation = activation if activation is not None else SiLU()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        self.input_blocks = nn.ModuleList(
            [
                torch.nn.Sequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        input_block_sizes = [spatial_size]
        ch = model_channels
        ds = 1
        current_sp = spatial_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    MyResBlock(
                        ch,
                        current_sp,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        activation=self.activation,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(torch.nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    torch.nn.Sequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                input_block_sizes.append(current_sp)
                ds *= 2
                current_sp = (current_sp + 1) // 2

        self.middle_block = torch.nn.Sequential(
            MyResBlock(
                ch,
                current_sp,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                activation=self.activation,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            MyResBlock(
                ch,
                current_sp,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                activation=self.activation,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    MyResBlock(
                        ch + input_block_chans.pop(),
                        current_sp,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        activation=self.activation,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                    current_sp = input_block_sizes.pop()
                self.output_blocks.append(torch.nn.Sequential(*layers))

        self.out = nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=[current_sp, current_sp, current_sp]),
            self.activation,
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return torch.float32  # FIXED
        # return next(self.input_blocks.parameters()).dtype

    def forward(self, x, y=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param low_cond: an [N x C x ...]  Tensor of condition.
        :return: an [N x C x ...] Tensor of outputs.
        """

        ## concat the condition
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.output_blocks:

            # handling for non-even inputs
            # h = F.interpolate(h, size= hs[-1].size()[-3:], mode='trilinear')
            if hs[-1].size(-1) < h.size(-1):
                h = h[..., :-1]
            if hs[-1].size(-2) < h.size(-2):
                h = h[..., :-1, :]
            if hs[-1].size(-3) < h.size(-3):
                h = h[..., :-1, :, :]

            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w l -> b (h w l) c")
        k = rearrange(k, "b c h w l-> b c (h w l)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w l -> b c (h w, l)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w l) -> b c h w l", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=False,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        # print(x.type(), context.type(), self.norm1)
        x = self.attn1(self.norm1(x.float())) + x
        x = self.attn2(self.norm2(x.float()), context=context.float()) + x.float()
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels)

        self.proj_in = nn.Conv3d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(
            nn.Conv3d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        # print(x.shape, context.shape)
        b, c, h, w, l = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w l -> b (h w l) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w l) c -> b c h w l", h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
