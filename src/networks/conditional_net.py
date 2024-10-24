import time

import numpy as np
import torch
import torch as th
from torch import nn

from src.experiments.utils.wavelet_utils import WaveletData
from src.networks.diffusion_modules.dwt import DWTInverse3d
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
from src.networks.diffusion_modules.nn import (
    SiLU,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
)
from src.networks.diffusion_modules.resample import (
    UniformSampler,
    LossSecondMomentResampler,
)
from src.networks.diffusion_modules.sparse_network import SparseComposer
from src.networks.diffusion_network import (
    TimestepEmbedSequential,
    ResBlock,
    Upsample,
    Downsample,
    zero_module,
    SpatialTransformer,
)
from src.networks.uvit_utils import Condition_UVIT


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


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


def remove_duplicate(tensor_1, tensor_2):
    # borrow from https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
    combined = torch.cat((tensor_1, tensor_2), dim=0)
    uniques, counts = combined.unique(return_counts=True, dim=0)
    difference = uniques[counts == 1]

    return difference


class get_model_progressive(nn.Module):
    def __init__(self, args):
        super(get_model_progressive, self).__init__()

        ### Local Model Hyperparameters
        self.args = args

        ### the networks
        if self.use_condition():
            if (
                hasattr(self.args, "use_wavelet_conditions")
                and self.args.use_wavelet_conditions
            ):
                context_emb_dim = self.args.ae_z_dim
            elif (
                hasattr(self.args, "use_pointcloud_conditions")
                and self.args.use_pointcloud_conditions
            ):
                context_emb_dim = self.args.pc_output_dim
            elif (
                hasattr(self.args, "use_voxel_conditions")
                and self.args.use_voxel_conditions
            ):
                context_emb_dim = self.args.voxel_context_dim
            elif (
                hasattr(self.args, "use_multiple_views_inferences")
                and self.args.use_multiple_views_inferences
            ):
                context_emb_dim = self.args.cond_grid_emb_size
            elif (
                hasattr(self.args, "use_multiple_views_grids")
                and self.args.use_multiple_views_grids
            ):
                context_emb_dim = self.args.cond_grid_emb_size
            elif hasattr(self.args, "use_all_views") and self.args.use_all_views:
                context_emb_dim = self.args.cond_emb_dim
            else:
                context_emb_dim = self.args.cond_grid_emb_size

            ## Mapping network
            self.mapping_network = nn.ModuleList([])
            if args.num_mapping_layers == 0:
                self.mapping_network.append(nn.Identity())
            else:
                for _ in range(args.num_mapping_layers):
                    if args.cond_mapping_type == "with_layernorm":
                        self.mapping_network.append(
                            nn.Sequential(
                                nn.Linear(context_emb_dim, context_emb_dim),
                                nn.LayerNorm(context_emb_dim),
                                nn.LeakyReLU(),
                            )
                        )
                    else:
                        self.mapping_network.append(
                            nn.Sequential(
                                nn.Linear(context_emb_dim, context_emb_dim),
                                nn.LeakyReLU(),
                            )
                        )

            ## positional encoding for network
            if (
                hasattr(self.args, "use_wavelet_conditions")
                and self.args.use_wavelet_conditions
            ):
                pass
            elif (
                hasattr(self.args, "use_pointcloud_conditions")
                and self.args.use_pointcloud_conditions
            ):
                if (
                    not hasattr(self.args, "use_pointvoxel_encoder")
                    or not self.args.use_pointvoxel_encoder
                ):
                    self.cond_pos_emb = nn.init.trunc_normal_(
                        nn.Parameter(
                            torch.zeros(1, self.args.num_inds, context_emb_dim)
                        ),
                        0.0,
                        0.02,
                    )
            elif (
                hasattr(self.args, "use_voxel_conditions")
                and self.args.use_voxel_conditions
            ):
                self.cond_pos_emb = nn.init.trunc_normal_(
                    nn.Parameter(
                        torch.zeros(
                            1,
                            self.args.voxel_dim
                            * self.args.voxel_dim
                            * self.args.voxel_dim,
                            context_emb_dim,
                        )
                    ),
                    0.0,
                    0.02,
                )
            elif (
                not hasattr(self.args, "use_all_views")
                or not self.args.use_all_views
                or (
                    hasattr(self.args, "use_multiple_views_inferences")
                    and self.args.use_multiple_views_inferences
                )
            ) or (
                hasattr(self.args, "use_multiple_views_grids")
                and self.args.use_multiple_views_grids
            ):
                self.cond_pos_emb = nn.init.trunc_normal_(
                    nn.Parameter(torch.zeros(1, args.cond_grid_size, context_emb_dim)),
                    0.0,
                    0.02,
                )

            if self.args.dp_cond_type in ["learnable"]:
                self.cond_zero_emb = nn.init.trunc_normal_(
                    nn.Parameter(torch.zeros(1, args.cond_grid_size, context_emb_dim)),
                    0.0,
                    0.02,
                )

            ### network parameters for
            if hasattr(self.args, "use_camera_index") and self.args.use_camera_index:
                if (
                    hasattr(self.args, "use_multiple_views_inferences")
                    and self.args.use_multiple_views_inferences
                ) or (
                    hasattr(self.args, "use_multiple_views_grids")
                    and self.args.use_multiple_views_grids
                ):
                    camera_emb_dim = self.args.cond_grid_emb_size
                elif hasattr(self.args, "use_all_views") and self.args.use_all_views:
                    camera_emb_dim = self.args.cond_emb_dim
                else:
                    camera_emb_dim = self.args.cond_grid_emb_size
                self.camera_emb = nn.Embedding(self.args.max_images_num, camera_emb_dim)

        ### sparse models
        self.dwt_sparse_composer = SparseComposer(
            input_shape=[args.resolution, args.resolution, args.resolution],
            J=args.max_depth,
            wave=args.wavelet,
            mode=args.padding_mode,
            inverse_dwt_module=None,
        )

        ### network setting
        if not self.use_condition():
            context_dim = None
            use_single_context = False
            context_emb_dim = None
        elif (
            self.args.use_all_views and self.args.input_view_cnt == 1
        ):  ### if single view clip features
            context_dim = None
            use_single_context = True
        else:
            context_dim = context_emb_dim
            use_single_context = False

        high_size = (
            511
            if not hasattr(self.args, "max_training_level")
            or self.args.max_training_level == self.args.max_depth
            else (2**3) ** self.args.max_training_level - 1
        )

        input_size = (
            self.args.profile_sizes[1]
            if hasattr(self.args, "profile_sizes") and self.args.profile_unet
            else 46
        )
        self.unet = Condition_UVIT(
            in_channels=1 + high_size,
            model_channels=self.args.unet_model_channels,
            out_channels=1 + high_size,
            num_res_blocks=self.args.unet_num_res_blocks,
            channel_mult=self.args.unet_channel_mult,
            input_size=input_size,
            use_scale_shift_norm=True,
            dropout=0,
            dims=3,
            activation=None,
            context_dim=context_dim,
            context_emb_dim=context_emb_dim,
            use_single_context=use_single_context,
            num_transformer_blocks=args.num_transformer_blocks,
            with_self_att=(
                args.with_self_att if hasattr(args, "with_self_att") else False
            ),
            add_num_register=(
                args.add_num_register if hasattr(args, "add_num_register") else 0
            ),
            learnable_skip_r=(
                args.learnable_skip_r if hasattr(args, "learnable_skip_r") else None
            ),
            add_condition_time_ch=(
                args.add_condition_time_ch
                if hasattr(args, "add_condition_time_ch")
                else None
            ),
            add_condition_input_ch=(
                args.add_condition_input_ch
                if hasattr(args, "add_condition_input_ch")
                else None
            ),
            no_cross_attention=(
                True
                if hasattr(args, "use_wavelet_conditions")
                and self.args.use_wavelet_conditions
                else False
            ),
        )

        if hasattr(self.args, "profile_unet") and self.args.profile_unet:
            from torchsummary import summary

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.unet = self.unet.to(device)
            summary(self.unet, input_size=tuple(self.args.profile_sizes), device="cuda")
            exit(0)

        self.dwt_inverse_3d = DWTInverse3d(
            J=args.max_depth, wave=args.wavelet, mode=args.padding_mode
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

        ## avg + scale values
        low_avg = self.args.low_avg if hasattr(self.args, "low_avg") else 2.20
        print(f"Low avg used : {low_avg}")
        self.avg_value = torch.from_numpy(
            np.array([low_avg] + [0] * high_size)
        )  ### HARD_CODE first

        if hasattr(self.args, "use_normalize_std") and self.args.use_normalize_std:
            assert self.args.std is not None
            self.scale_value = torch.from_numpy(np.array(self.args.std))
        else:
            self.scale_value = torch.ones_like(self.avg_value)

    def reset_diffusion_module(self):
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

    def process_condition(self, condition, image_index=None):

        ### return None for condition
        if condition is None:
            return None

        #### Condition embeddings
        for layer in self.mapping_network:
            condition = layer(condition)

        if (
            not hasattr(self.args, "use_all_views") or not self.args.use_all_views
        ) and hasattr(self, "cond_pos_emb"):
            condition = condition + self.cond_pos_emb

        if (
            hasattr(self.args, "use_multiple_views_inferences")
            and self.args.use_multiple_views_inferences
        ):
            condition = condition + self.cond_pos_emb.unsqueeze(0)

        if (
            hasattr(self.args, "use_multiple_views_grids")
            and self.args.use_multiple_views_grids
        ):
            condition = condition + self.cond_pos_emb.unsqueeze(0)

        if image_index is not None:
            camera_emb = self.camera_emb(image_index)
            if (
                hasattr(self.args, "use_multiple_views_grids")
                and self.args.use_multiple_views_grids
            ):
                camera_emb = camera_emb.unsqueeze(2)
            else:
                if len(camera_emb.size()) != len(
                    condition.size()
                ):  # unsqueeze when using grid features
                    camera_emb = camera_emb.unsqueeze(1)
            condition = condition + camera_emb

        if (
            hasattr(self.args, "use_multiple_views_grids")
            and self.args.use_multiple_views_grids
        ):
            condition = condition.view(condition.size(0), -1, condition.size(-1))

        return condition

    def inference_completion(self, wavelet_volume_input, mask):

        assert not self.use_condition()

        ## compute inference shape
        shape = wavelet_volume_input.size()

        condition = None

        condition = self.process_condition(condition)

        model_kwargs = {"latent_codes": condition}

        ## shfit the mean
        if self.args.use_shift_mean:
            avg_value = (
                self.avg_value.unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .to(wavelet_volume_input.device)
            )
            avg_value = avg_value.type(wavelet_volume_input.dtype)
            wavelet_volume_input = wavelet_volume_input - avg_value

        ## normalize the wavelet volume
        if self.args.use_normalize_std:
            scale_value = (
                self.scale_value.unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .to(wavelet_volume_input.device)
            )
            scale_value = scale_value.type(wavelet_volume_input.dtype)
            wavelet_volume_input = wavelet_volume_input / scale_value

        ## Additional model kwargs
        model_kwargs["mask"] = mask
        model_kwargs["wavelet_volume"] = wavelet_volume_input
        model_kwargs["sampling_steps"] = self.args.sampling_steps
        model_kwargs["jumpy_steps"] = self.args.jumpy_steps

        ## inference
        pred, _ = self.inference_diffusion_module.p_sample_loop_completion(
            model=self.unet,
            shape=shape,
            device=wavelet_volume_input.device,
            clip_denoised=False,
            progress=True,
            model_kwargs=model_kwargs,
        )

        if self.args.use_normalize_std:
            scale_value = (
                self.scale_value.unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .to(pred.device)
            )
            scale_value = scale_value.type(pred.dtype)
            pred = pred * scale_value

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

        order = (2**self.args.max_training_level) ** 3
        idx = order - 1
        pred[:, 1 : pred.size(1) - idx, :, :, :] = 0  # filter other stuff

        wavelet_data = WaveletData(
            shape_list=self.dwt_sparse_composer.shape_list,
            output_stage=self.args.max_training_level,
            max_depth=self.args.max_depth,
            wavelet_volume=pred[0:1],
        )
        low_pred, highs_pred = wavelet_data.convert_low_highs()

        return low_pred, highs_pred

    def inference(
        self,
        low,
        condition,
        img_idx,
        local_rank,
        current_stage,
        return_wavelet_volume=False,
        progress=True,
    ):
        ## compute inference
        shape = (low.size(0), 512, low.size(2), low.size(3), low.size(4))

        if (
            hasattr(self.args, "max_training_level")
            and self.args.max_training_level != self.args.max_depth
        ):
            high_size = (2**3) ** self.args.max_training_level - 1
            shape = (low.size(0), 1 + high_size, low.size(2), low.size(3), low.size(4))

        condition_before_process = condition

        ### adding masking features
        if hasattr(self.args, "use_mask_inference") and self.args.use_mask_inference:
            condition[:, 1:] = 0  # masking

        condition = self.process_condition(condition, img_idx)

        if (
            self.args.use_all_views and self.args.input_view_cnt == 1
        ):  ### if single view clip features
            assert condition.size(1) == 1
            condition = condition.squeeze(1)
            model_kwargs = {"single_context": condition}  # only one latent ccode
        else:
            model_kwargs = {"latent_codes": condition}

        if self.args.dp_cond is not None:
            if self.args.dp_cond_type is not None:
                if len(condition.size()) == 3:
                    condition_zero = self.cond_zero_emb.repeat(condition.size(0), 1, 1)
                elif len(condition.size()) == 4:
                    condition_zero = self.cond_zero_emb.repeat(
                        condition.size(0), condition.size(1), 1, 1
                    )
                else:
                    raise Exception("Unknown Condition.....")
            else:
                condition_zero = torch.zeros_like(condition_before_process)
            condition_zero = self.process_condition(condition_zero, img_idx)
            model_kwargs["condition_zero"] = condition_zero
            model_kwargs["guidance_scale"] = self.args.scale
            model_kwargs["dp_cond"] = self.args.dp_cond

            ### multiple view inferences
            if (
                hasattr(self.args, "use_multiple_views_inferences")
                and self.args.use_multiple_views_inferences
            ):
                model_kwargs["use_multiple_views_inferences"] = (
                    self.args.use_multiple_views_inferences
                )

            if hasattr(self.args, "use_eps_dp") and self.args.use_eps_dp:
                model_kwargs["use_eps_dp"] = self.args.use_eps_dp

        pred, _ = self.inference_diffusion_module.p_sample_loop(
            model=self.unet,
            shape=shape,
            device=low.device,
            clip_denoised=False,
            progress=progress,
            model_kwargs=model_kwargs,
        )

        if hasattr(self.args, "use_normalize_std") and self.args.use_normalize_std:
            scale_value = (
                self.scale_value.unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .to(pred.device)
            )
            scale_value = scale_value.type(pred.dtype)
            pred = pred * scale_value

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

        wavelet_data = WaveletData(
            shape_list=self.dwt_sparse_composer.shape_list,
            output_stage=self.args.max_training_level,
            max_depth=self.args.max_depth,
            wavelet_volume=pred[local_rank : local_rank + 1],
        )
        low_pred, highs_pred = wavelet_data.convert_low_highs()

        if return_wavelet_volume:
            return low_pred, highs_pred, pred[local_rank : local_rank + 1]
        else:
            return low_pred, highs_pred

    def training_losses(
        self,
        low,
        high_indices,
        high_values,
        condition,
        current_stage,
        img_idx,
        high_indices_empty=None,
        high_values_mask=None,
    ):

        # loss computation
        batch_size = low.size(0)
        t, weights = self.sampler.sample(batch_size, device=low.device)

        if self.args.dp_cond is not None:
            anti_mask = prob_mask_like(
                (condition.size(0)), 1 - self.args.dp_cond, condition.device
            )
            anti_mask = anti_mask.unsqueeze(1).unsqueeze(1)

            ### fixing conditioning
            if len(condition.size()) > len(anti_mask.size()):
                anti_mask = anti_mask.unsqueeze(1)

            if self.args.dp_cond_type is not None:
                condition = torch.where(anti_mask, condition, self.cond_zero_emb)
            else:
                condition = condition * anti_mask

        condition = self.process_condition(condition, img_idx)

        wavelet_data = WaveletData(
            shape_list=self.dwt_sparse_composer.shape_list,
            output_stage=self.args.max_training_level,
            max_depth=self.args.max_depth,
            low=low,
            highs_indices=high_indices,
            highs_values=high_values,
        )
        data_samples = wavelet_data.convert_wavelet_volume()

        model_kawrgs = {}

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

        if self.args.use_normalize_std:
            scale_value = (
                self.scale_value.unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .to(data_samples.device)
            )
            scale_value = scale_value.type(data_samples.dtype)
            data_samples = data_samples / scale_value
            model_kawrgs["use_normalize_std"] = True

        model_kawrgs["current_stage"] = current_stage

        if self.args.use_sample_training:
            assert high_indices_empty is not None and high_values_mask is not None

            ## compute training indices
            non_empty_indices = high_indices[:, :, 1:].long()
            training_indices = torch.cat((non_empty_indices, high_indices_empty), dim=1)
            training_indices = pad_with_batch_idx(training_indices)

            model_kawrgs["training_indices"] = training_indices

            high_values_mask = high_values_mask.repeat(1, 2)
            model_kawrgs["high_values_mask"] = high_values_mask

        model_kawrgs["no_rebalance_loss"] = (
            hasattr(self.args, "no_rebalance_loss") and self.args.no_rebalance_loss
        )
        model_kawrgs["use_mse_loss"] = (
            hasattr(self.args, "use_mse_loss") and self.args.use_mse_loss
        )

        if (
            self.args.use_all_views and self.args.input_view_cnt == 1
        ):  ### if single view clip features
            assert condition.size(1) == 1
            latent_codes = None
            single_context = condition.squeeze(1)
        else:
            latent_codes = condition
            single_context = None

        loss = self.diffusion_module.training_losses(
            self.unet,
            x_start=data_samples,
            t=t,
            model_kwargs=model_kawrgs,
            latent_codes=latent_codes,
            single_context=single_context,
        )

        ### update the loss records
        if self.args.diffusion_sampler == "second-order":
            if hasattr(self.args, "use_loss_resample") and self.args.use_loss_resample:
                self.sampler.update_with_local_losses(t, loss["loss"])

        losses = {}
        losses["loss"] = torch.mean(weights * loss["loss"])
        losses["base_loss"] = torch.mean(weights * loss["base_loss"])
        for idx in range(current_stage):
            if f"loss_{idx + 1}" in loss:
                losses[f"loss_{idx + 1}"] = torch.mean(
                    weights * loss[f"loss_{idx + 1}"]
                )

        ## new logging with timestep quartile considered
        for quartile in range(4):
            quartile_mask = t // (self.diffusion_module.num_timesteps // 4) == quartile
            losses[f"base_loss_q{quartile}"] = loss["base_loss"][quartile_mask].mean()
            for idx in range(current_stage):
                if f"loss_{idx + 1}" in loss:
                    losses[f"loss_{idx + 1}_q{quartile}"] = loss[f"loss_{idx + 1}"][
                        quartile_mask
                    ].mean()

        return losses

    def use_condition(self):
        return (
            self.args.use_pointcloud_conditions
            or self.args.use_voxel_conditions
            or self.args.use_image_conditions
            or (
                hasattr(self.args, "use_wavelet_conditions")
                and self.args.use_wavelet_conditions
            )
        )


############################################## NETWROK UTILS ##############################################


class Condition_UNetModel(nn.Module):
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
        context_dim=None,
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
                        SpatialTransformer(
                            ch,
                            num_heads,
                            ch,
                            depth=1,
                            dropout=0.0,
                            context_dim=context_dim,
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
            SpatialTransformer(
                ch, num_heads, ch, depth=1, dropout=0.0, context_dim=context_dim
            ),
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
                        SpatialTransformer(
                            ch,
                            num_heads,
                            ch,
                            depth=1,
                            dropout=0.0,
                            context_dim=context_dim,
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

    def forward(self, x, timesteps, y=None, low_cond=None, latent_codes=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param low_cond: an [N x C x ...]  Tensor of condition.
        :return: an [N x C x ...] Tensor of outputs.
        """

        # print("aa",x.shape, latent_codes.shape, y, low_cond, timesteps)
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
            h = module(h, emb, context=latent_codes.type(h.type()))
            hs.append(h)

        h = self.middle_block(h, emb, context=latent_codes.type(h.type()))
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
            h = module(cat_in, emb, context=latent_codes.type(h.type()))

        # h = h.type(x.dtype)
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


################################################### Attention Stuff ############################################################


# feedforward
