import os

# import math
import mcubes
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from shutil import copyfile


from src.clip_mod import get_clip_model, tokenize

# from utils import experimenter
from src.dataset_utils import get_image_transform
from src.experiments.utils.wavelet_utils import (
    extract_highs_from_values,
    extract_full_indices,
    WaveletData,
)
from src.networks import diffusion_network, conditional_net
from src.networks.points_network import PointNet_Head
from src.networks.voxels_network import Encoder_Down_4
from src.networks.diffusion_modules.network_ae import WaveletEncoder
from src.networks.diffusion_modules.point_voxels import PointVoxelEncoder

from src.utils import visualization


class Trainer_Diffusion_Progressive_Network(pl.LightningModule):
    def __init__(self, args, diffusion_rescale_timestep=None, scale=None):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        if diffusion_rescale_timestep is not None:
            self.args.diffusion_rescale_timestep = diffusion_rescale_timestep
        if scale is not None:
            self.args.scale = scale
        if (
            hasattr(self.args, "use_image_conditions")
            and self.args.use_image_conditions
        ):
            self.image_transform = get_image_transform(args)

        if (
            hasattr(self.args, "use_image_conditions")
            and self.args.use_image_conditions
        ):
            ### setup the things for CLIP
            args, clip_model, clip_preprocess = get_clip_model(args)
            args.condition_dim = args.cond_grid_emb_size
            args.num_cond_vectors = args.cond_grid_size

            if (
                not hasattr(self.args, "use_image_features_only")
                or not self.args.use_image_features_only
                or (self.args.use_all_views and self.args.input_view_cnt == 1)
            ):
                self.clip_model = clip_model
                for param in self.clip_model.parameters():
                    param.requires_grad = False

        self.network = conditional_net.get_model_progressive(args)

        self.current_stage = self.args.start_stage

        if hasattr(self.args, "max_training_level"):
            self.current_stage = min(self.args.max_training_level, self.current_stage)

        ### condition encoder
        if (
            hasattr(self.args, "use_wavelet_conditions")
            and self.args.use_wavelet_conditions
        ):
            high_size = (
                511
                if not hasattr(self.args, "max_training_level")
                or self.args.max_training_level == self.args.max_depth
                else (2**3) ** self.args.max_training_level - 1
            )
            self.encoder = WaveletEncoder(ae_input_channel=1 + high_size, args=args)
        if (
            hasattr(self.args, "use_pointcloud_conditions")
            and self.args.use_pointcloud_conditions
        ):
            if (
                hasattr(self.args, "use_pointvoxel_encoder")
                and self.args.use_pointvoxel_encoder
            ):
                self.encoder = PointVoxelEncoder(args)
            else:
                self.encoder = PointNet_Head(
                    output_dim=self.args.pc_output_dim,
                    pc_dims=self.args.pc_dims,
                    num_inds=self.args.num_inds,
                    num_heads=self.args.num_heads,
                )
        elif (
            hasattr(self.args, "use_voxel_conditions")
            and self.args.use_voxel_conditions
        ):
            self.encoder = Encoder_Down_4(
                channel_in=1, channel_out=self.args.voxel_context_dim
            )

        self.ema_state_dict = None

    def on_load_checkpoint(self, checkpoint):
        # lightning function
        self.ema_state_dict = checkpoint["state_dict_ema"]

    def test_step(self, data, batch_idx):

        if self.args.train_mode == "complete":
            self.test_completion_sample(data)
        else:
            batch_size = data["low"].size(0)

            for idx in range(batch_size):

                for i in range(self.args.test_trials):
                    posfix = f"_{i}" if self.args.test_trials > 1 else ""
                    id = data["id"][idx]
                    with torch.no_grad():
                        low_pred, highs_pred, wavelet_volume = self.inference_sample(
                            data, idx, return_wavelet_volume=True
                        )

                    # saving obj
                    obj_path = os.path.join(self.args.generate_dir, f"{id}{posfix}.obj")
                    self.save_visualization_obj(
                        obj_path=obj_path, samples=(low_pred, highs_pred)
                    )

                    # convert gt
                    low_data = data["low"]
                    highs_indices_data = data["high_indices"]
                    highs_values_data = data["high_values"]

                    wavelet_data = WaveletData(
                        shape_list=self.network.dwt_sparse_composer.shape_list,
                        output_stage=self.args.max_training_level,
                        max_depth=self.args.max_depth,
                        low=low_data[idx : idx + 1],
                        highs_indices=highs_indices_data[idx : idx + 1],
                        highs_values=highs_values_data[idx : idx + 1],
                    )
                    low_gt, highs_gt = wavelet_data.convert_low_highs()

                    ### output obj
                    obj_path = os.path.join(self.args.generate_dir, f"{id}_gt.obj")
                    self.save_visualization_obj(obj_path, (low_gt, highs_gt))

                    if (
                        hasattr(self.args, "use_image_conditions")
                        and self.args.use_image_conditions
                    ):
                        if len(data["images"].size()) > 4:
                            for img_idx in range(data["images"].size(1)):
                                image_path = os.path.join(
                                    self.args.generate_dir,
                                    f"{id}_{img_idx}_input.png",
                                )
                                # image_data = self.extract_images(data, img_idx=img_idx)
                                # visualization.save_normalized_image_matplotlib(
                                #     image_data[idx].permute(1, 2, 0).detach().cpu().numpy(),
                                #     file_name=image_path)
                                Image.fromarray(
                                    data["images"][idx, img_idx].cpu().numpy()
                                ).save(image_path)
                        else:
                            image_path = os.path.join(
                                self.args.generate_dir, f"{id}_input.png"
                            )
                            # image_data = self.extract_images(data)
                            # visualization.save_normalized_image_matplotlib(image_data[idx].permute(1, 2, 0).detach().cpu().numpy(),
                            #                                                file_name=image_path)
                            Image.fromarray(data["images"][idx].cpu().numpy()).save(
                                image_path
                            )

                    if (
                        hasattr(self.args, "use_voxel_conditions")
                        and self.args.use_voxel_conditions
                    ):
                        voxel_data = data["voxels"].cpu().detach().numpy()[idx, 0]
                        voxel_image_path = os.path.join(
                            self.args.generate_dir, f"{id}_voxel.png"
                        )
                        visualization.save_voxel_image(voxel_data, voxel_image_path)

                        # save voxels
                        self.save_voxels(
                            os.path.join(
                                self.args.generate_dir,
                                f"{id}_{self.args.voxel_resolution}.npz",
                            ),
                            voxel_data,
                        )

                    if (
                        hasattr(self.args, "use_pointcloud_conditions")
                        and self.args.use_pointcloud_conditions
                    ):
                        pointcloud_data = data["points"].cpu().detach().numpy()[idx]
                        pointcloud_ply_path = os.path.join(
                            self.args.generate_dir, f"{id}_pointcloud.ply"
                        )
                        visualization.save_point_cloud_as_ply(
                            pointcloud_ply_path, pointcloud_data
                        )

                        # save points
                        np.save(
                            os.path.join(
                                self.args.generate_dir,
                                f"{id}_{pointcloud_data.shape[0]}.npy",
                            ),
                            pointcloud_data,
                        )

                    ### saving the wavelet volume
                    wavelet_volume_pred = wavelet_volume.detach().cpu().numpy()[0]
                    wavelet_volume_path = os.path.join(
                        self.args.generate_dir, f"{id}{posfix}_wavelet.npy"
                    )
                    np.save(wavelet_volume_path, wavelet_volume_pred)

    ############################################## INPUT FEATURE EXTRACTION ##############################################
    # function for extracting features from inputs
    def extract_input_features(self, data, data_type, is_train=False, to_cuda=False):

        ### convert to cuda tensor
        if to_cuda:
            for key, item in data.items():
                if torch.is_tensor(item):
                    data[key] = item.to(self.device)

        assert data_type in ["wavelets", "points", "voxels", "images", "texts"]
        if data_type == "wavelets":
            low = data["low"]
            high_indices = data["high_indices"]
            high_values = data["high_values"]
            wavelet_data = WaveletData(
                shape_list=self.network.dwt_sparse_composer.shape_list,
                output_stage=self.args.max_training_level,
                max_depth=self.args.max_depth,
                low=low,
                highs_indices=high_indices,
                highs_values=high_values,
            )
            wavelet_inputs = wavelet_data.convert_wavelet_volume()
            input_features = self.encoder(wavelet_inputs)
        elif data_type == "points":
            points_inputs = data["points"]
            input_features = self.encoder(points_inputs)
        elif data_type == "voxels":
            voxels_inputs = data["voxels"]
            input_features = self.encoder(voxels_inputs)
            input_features = torch.permute(input_features, (0, 2, 3, 4, 1))
            input_features = input_features.view(
                (
                    input_features.size(0),
                    input_features.size(1)
                    * input_features.size(2)
                    * input_features.size(3),
                    input_features.size(4),
                )
            )
        elif data_type == "images":
            if (
                hasattr(self.args, "use_image_features_only")
                and self.args.use_image_features_only
            ):
                input_features = (
                    data["image_features"].type(torch.FloatTensor).to(self.device)
                )
                ## noising if needed
                if (
                    hasattr(self.args, "use_noised_clip_features")
                    and self.args.use_noised_clip_features
                    and is_train
                ):
                    input_features = get_noised_image_features(
                        input_features, self.args.noise_cond_para
                    )
                input_features = input_features / input_features.norm(
                    dim=-1, keepdim=True
                )
            else:
                image = data["images"].type(torch.FloatTensor).to(self.device)
                input_features = self.extract_input_image_features(
                    image, is_train=is_train
                )
        elif data_type == "texts":
            texting_texts = [
                "a truck",
                "a jet plane",
                "a dog",
                "an eiffel tower",
                "a human sitting down",
                "a round chair",
                "a guitar",
                "a screw",
            ]
            text = tokenize(texting_texts)
            text_features = self.clip_model.encode_text(text.to(self.device))
            text_features = text_features.type(torch.FloatTensor).to(self.device)
            input_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            raise Exception("Unknown features....")

        return input_features

    def extract_images(self, data, img_idx=0):
        image = data["images"].type(torch.FloatTensor).to(self.device)
        if len(image.size()) > 4:
            # return the image of img_idx
            image_data = image[:, img_idx]
        else:
            image_data = image
        return image_data

    def extract_img_idx(self, data, data_idx=None):
        if hasattr(self.args, "use_camera_index") and self.args.use_camera_index:
            img_idx = data["img_idx"].type(torch.LongTensor).to(self.device)
            if data_idx is not None:
                img_idx = img_idx[data_idx : data_idx + 1]
        else:
            img_idx = None
        return img_idx

    def extract_input_image_features(self, image, is_train=False):
        with torch.no_grad():
            if hasattr(self.args, "use_all_views") and self.args.use_all_views:
                batch_size = image.size(0)
                image_reshaped = image.view(
                    batch_size * image.size(1),
                    image.size(2),
                    image.size(3),
                    image.size(4),
                )
                image_features, image_features_grids = (
                    self.clip_model.get_image_features(image_reshaped)
                )

                if (
                    hasattr(self.args, "use_multiple_views_inferences")
                    and self.args.use_multiple_views_inferences
                ) or (
                    hasattr(self.args, "use_multiple_views_grids")
                    and self.args.use_multiple_views_grids
                ):
                    input_features = image_features_grids[-1].view(
                        batch_size, image.size(1), -1, image_features_grids[-1].size(-1)
                    )
                    input_features = input_features.detach()
                else:
                    ## noising if needed
                    if (
                        hasattr(self.args, "use_noised_clip_features")
                        and self.args.use_noised_clip_features
                        and is_train
                    ):
                        image_features = get_noised_image_features(
                            image_features, self.args.noise_cond_para
                        )

                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    image_features = image_features.view(batch_size, image.size(1), -1)
                    input_features = image_features.detach()
            else:
                image_features, image_features_grids = (
                    self.clip_model.get_image_features(image)
                )
                input_features = image_features_grids[-1].detach()
            return input_features.float()

    ############################################## INFERENCES ##############################################
    def inference_sample(
        self, data, data_idx, return_wavelet_volume=False, progress=True
    ):
        low_data = data["low"].type(torch.FloatTensor).to(self.device)
        if (
            hasattr(self.args, "use_wavelet_conditions")
            and self.args.use_wavelet_conditions
        ):
            input_features = self.extract_input_features(
                data, data_type="wavelets", is_train=False, to_cuda=True
            )
            outputs = self.network.inference(
                low_data[data_idx : data_idx + 1],
                input_features[data_idx : data_idx + 1],
                None,
                local_rank=0,
                current_stage=self.current_stage,
                return_wavelet_volume=return_wavelet_volume,
                progress=progress,
            )
        elif (
            hasattr(self.args, "use_pointcloud_conditions")
            and self.args.use_pointcloud_conditions
        ):
            input_features = self.extract_input_features(
                data, data_type="points", is_train=False, to_cuda=True
            )
            outputs = self.network.inference(
                low_data[data_idx : data_idx + 1],
                input_features[data_idx : data_idx + 1],
                None,
                local_rank=0,
                current_stage=self.current_stage,
                return_wavelet_volume=return_wavelet_volume,
                progress=progress,
            )
        elif (
            hasattr(self.args, "use_voxel_conditions")
            and self.args.use_voxel_conditions
        ):
            input_features = self.extract_input_features(
                data, data_type="voxels", is_train=False, to_cuda=True
            )
            outputs = self.network.inference(
                low_data[data_idx : data_idx + 1],
                input_features[data_idx : data_idx + 1],
                None,
                local_rank=0,
                current_stage=self.current_stage,
                return_wavelet_volume=return_wavelet_volume,
                progress=progress,
            )
        elif self.args.use_all_views and self.args.input_view_cnt == 1:
            assert self.args.use_camera_index == False
            input_features = self.extract_input_features(
                data, data_type="texts", is_train=False, to_cuda=True
            )
            data_idx = self.trainer.global_rank % input_features.size(0)
            outputs = self.network.inference(
                low_data[data_idx : data_idx + 1],
                input_features[data_idx : data_idx + 1].unsqueeze(1),
                None,
                local_rank=0,
                current_stage=self.current_stage,
                return_wavelet_volume=return_wavelet_volume,
                progress=progress,
            )
        elif self.args.use_image_conditions:
            input_features = self.extract_input_features(
                data, data_type="images", is_train=False, to_cuda=True
            )
            img_idx = self.extract_img_idx(data, data_idx=data_idx)

            outputs = self.network.inference(
                low_data[data_idx : data_idx + 1],
                input_features[data_idx : data_idx + 1],
                img_idx,
                local_rank=0,
                current_stage=self.current_stage,
                return_wavelet_volume=return_wavelet_volume,
                progress=progress,
            )
        else:
            outputs = self.network.inference(
                low_data[data_idx : data_idx + 1],
                None,
                None,
                local_rank=0,
                current_stage=self.current_stage,
                return_wavelet_volume=return_wavelet_volume,
                progress=progress,
            )
        return outputs

    def test_completion_sample(self, data):
        wavelet_volume = data["wavelet_volume"]
        mask = data["mask"][0]
        save_id = data["id"][0]
        print(f"wavelet volume size : {wavelet_volume.size()}")
        print(f"mask size : {mask.size()}")

        # convert GT
        wavelet_data = WaveletData(
            shape_list=self.network.dwt_sparse_composer.shape_list,
            output_stage=self.args.max_training_level,
            max_depth=self.args.max_depth,
            wavelet_volume=wavelet_volume[0:1],
        )
        low_gt, highs_gt = wavelet_data.convert_low_highs()
        obj_path = os.path.join(self.args.generate_dir, f"{save_id}_gt.obj")
        self.save_visualization_obj(obj_path, (low_gt, highs_gt))

        ## save masked shape
        file_path = data["file_path"][0]
        masked_file_path = file_path.replace("_mask.npy", "_masked.obj")
        if os.path.exists(masked_file_path):
            copyfile(
                masked_file_path,
                os.path.join(self.args.generate_dir, f"{save_id}_masked.obj"),
            )

        for idx in range(self.args.completion_trials):
            if self.args.compile_model:
                self.network.inference_completion = torch.compile(
                    self.network.inference_completion
                )

            # obtain the low and high
            with torch.no_grad():
                low_pred, highs_pred = self.network.inference_completion(
                    wavelet_volume_input=wavelet_volume, mask=mask
                )

            obj_path = os.path.join(self.args.generate_dir, f"{save_id}_{idx}.obj")
            self.save_visualization_obj(obj_path, (low_pred, highs_pred))

    def set_inference_fusion_params(self, scale, diffusion_rescale_timestep):
        self.args.scale = scale
        self.args.diffusion_rescale_timestep = diffusion_rescale_timestep
        self.network.reset_diffusion_module()

    def test_inference(self, data, data_idx, save_dir, output_format="obj"):
        file_name = data["id"][data_idx]
        with torch.no_grad():
            low_pred, highs_pred = self.inference_sample(
                data, data_idx, return_wavelet_volume=False
            )

        if output_format == "sdf":
            sdf_path = os.path.join(save_dir, f"{file_name}.npz")
            self.save_sdf(sdf_path, (low_pred, highs_pred))
            return sdf_path
        else:
            obj_path = os.path.join(save_dir, f"{file_name}.obj")
            self.save_visualization_obj(
                obj_path=obj_path, samples=(low_pred, highs_pred)
            )
            return obj_path

    ############################################## HELP FUNCTIONS ##############################################

    def save_voxels(self, save_path, voxels):
        pts = np.argwhere(np.array(voxels) == 1)
        colors_dummy = np.zeros_like(pts)  # dummy color

        # save and upload
        np.savez_compressed(save_path, occupancy_arr=pts, color_numpy=colors_dummy)

    def save_sdf(self, save_path, samples):
        low, highs = samples
        sdf_recon = self.network.dwt_inverse_3d((low, highs))
        sdf_recon = sdf_recon.cpu().detach().numpy()[0, 0]
        np.savez(save_path, sdf=sdf_recon)

    def save_visualization_obj(self, obj_path, samples):
        low, highs = samples
        sdf_recon = self.network.dwt_inverse_3d((low, highs))
        vertices, triangles = mcubes.marching_cubes(
            sdf_recon.cpu().detach().numpy()[0, 0], 0.0
        )
        vertices = (vertices / self.args.resolution) * 2.0 - 1.0
        triangles = triangles[:, ::-1]
        mcubes.export_obj(vertices, triangles, obj_path)

    def get_parameters(self):
        parameters = list(self.network.parameters())
        if hasattr(self, "encoder"):
            parameters = parameters + list(self.encoder.parameters())
        return parameters

    def log_losses(self, losses, prefix=""):

        # process dict
        output = {
            f"{prefix}loss": losses["loss"],
            f"{prefix}base_loss": losses["base_loss"],
        }
        for idx in range(self.current_stage):
            if f"loss_{idx + 1}" in losses:
                output[f"{prefix}loss_{idx + 1}"] = losses[f"loss_{idx + 1}"]

        self.log(
            f"{prefix}loss",
            losses["loss"],
            sync_dist=True,
            on_step=True,
            rank_zero_only=True,
        )
        self.log(
            f"{prefix}base_loss",
            losses["base_loss"],
            sync_dist=True,
            on_step=True,
            rank_zero_only=True,
        )
        for idx in range(self.current_stage):
            if f"loss_{idx + 1}" in losses:
                self.log(
                    f"{prefix}loss_{idx + 1}",
                    losses[f"loss_{idx + 1}"],
                    sync_dist=True,
                    on_step=True,
                    rank_zero_only=True,
                )

        ### additional logging for quartile
        for quartile in range(4):
            if not torch.isnan(losses[f"base_loss_q{quartile}"]):
                self.log(
                    f"{prefix}base_loss_q{quartile}",
                    losses[f"base_loss_q{quartile}"],
                    sync_dist=True,
                    on_step=True,
                    rank_zero_only=True,
                )
            for idx in range(self.current_stage):
                if f"loss_{idx + 1}_q{quartile}" in losses and not torch.isnan(
                    losses[f"loss_{idx + 1}_q{quartile}"]
                ):
                    self.log(
                        f"{prefix}loss_{idx + 1}_q{quartile}",
                        losses[f"loss_{idx + 1}_q{quartile}"],
                        sync_dist=True,
                        on_step=True,
                        rank_zero_only=True,
                    )

        return output

    def extract_full_highs_gt(
        self, highs_full, highs_indices, highs_values, shape_list
    ):
        highs_full[
            0, highs_indices[:, 0], highs_indices[:, 1], highs_indices[:, 2], :
        ] = highs_values[0]
        highs_full = highs_full.reshape((-1, 511))  ## reshape the the correct size
        highs_full_indices = extract_full_indices(
            device=highs_full.device,
            max_depth=self.args.max_depth,
            shape_list=shape_list,
        )
        highs_full_recon = extract_highs_from_values(
            highs_full_indices, highs_full, self.args.max_depth, shape_list
        )
        return highs_full_recon


def get_noised_image_features(image_features, noise_cond_para):
    eps = 1e-7
    noise = torch.randn(
        image_features.size(0), image_features.size(1), image_features.size(2)
    ).to(image_features.device)
    noise = noise / (noise.norm(dim=-1, keepdim=True) + eps)
    image_features = (
        image_features
        + noise_cond_para * image_features.norm(dim=-1, keepdim=True) * noise
    )
    return image_features
