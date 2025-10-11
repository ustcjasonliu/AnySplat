from dataclasses import dataclass
from pathlib import Path
import os
import gc
import random
import shutil
from typing import Literal, Optional, Protocol, runtime_checkable, Any

import moviepy.editor as mpy
import torch
import torchvision
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
import torch.nn.functional as F

from lpips import LPIPS
from loss.loss_lpips import LossLpips
from loss.loss_mse import LossMse
from model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri


from ..loss.loss_distill import DistillLoss, huber_loss
from src.utils.render import generate_path
from src.utils.point import get_normal_map
from matplotlib import pyplot as plt
from model.encoder.encoder import Encoder, EncoderOutput
from src.model.decoder.decoder import DecoderOutput

from ..loss.loss_huber import HuberLoss, extri_intri_to_pose_encoding

# from model.types import Gaussians

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, abs_relative_difference, delta1_acc
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose, rotation_6d_to_matrix, align_and_transform
from ..misc.image_io import prep_image, save_image, save_video, save_interpolated_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    extrapolate_extrinsics,
    interpolate_extrinsics,
    interpolate_intrinsics,
    interpolate_trajectory
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
# from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .ply_export import export_ply, save_poses
from src.diffix3d.diffix_util import DiffixUtil
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import copy

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    generate_video: bool
    mode: Literal["inference", "evaluation"]
    image_folder: str


@dataclass
class TrainCfg:
    output_path: Path
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    pose_loss_alpha: float = 1.0
    pose_loss_delta: float = 1.0
    cxt_depth_weight: float = 0.01
    weight_pose: float = 1.0
    weight_depth: float = 1.0
    weight_normal: float = 1.0
    render_ba: bool = False
    render_ba_after_step: int = 0


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model: nn.Module
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        model: nn.Module,
        losses: list[Loss],
        step_tracker: StepTracker | None
    ) -> None:
        super().__init__()
        self._diffix_util = DiffixUtil()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        
        # Set up the model.
        self.encoder_visualizer = None
        self.model = model
        self.data_shim = get_data_shim(self.model.encoder)
        self.losses = nn.ModuleList(losses)
        
        if self.model.encoder.pred_pose:
            self.loss_pose = HuberLoss(alpha=self.train_cfg.pose_loss_alpha, delta=self.train_cfg.pose_loss_delta)
        
        if self.model.encoder.distill:
            self.loss_distill = DistillLoss(
                delta=self.train_cfg.pose_loss_delta,
                weight_pose=self.train_cfg.weight_pose,
                weight_depth=self.train_cfg.weight_depth,
                weight_normal=self.train_cfg.weight_normal
            )

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.detailed_save_interval = 100
        self.log_save_interval = 10
        self.ssim_loss_module = MS_SSIM(data_range=1.0, size_average=True)
        self.is_update_by_diffusion = False
        self.last_batch_context_extrinsics = None
        self.last_encoder_output_extrisics = None
        
    def on_train_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.datamodule.train_loader.dataset, "set_epoch"):
            self.trainer.datamodule.train_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.train_loader.sampler, "set_epoch"):
            self.trainer.datamodule.train_loader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        print(f"Validation epoch start on rank {self.trainer.global_rank}")
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.datamodule.val_loader.dataset, "set_epoch"):
            self.trainer.datamodule.val_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.val_loader.sampler, "set_epoch"):
            self.trainer.datamodule.val_loader.sampler.set_epoch(self.current_epoch)

    def save_depth_and_rgb(self, num_views, num_origin_views, video, depth, save_path):
        """Save the depth and rgb video to the save_path."""
     
        depth_norm = (depth - depth[::num_views].quantile(0.01)) / (
            depth[::num_views].quantile(0.99) - depth[::num_views].quantile(0.01)
        )
        depth_norm = plt.cm.turbo(depth_norm.cpu().detach().numpy())
        depth_colored = (
            torch.from_numpy(depth_norm[..., :3]).permute(0, 3, 1, 2).to(depth.device)
        )
        depth_colored = depth_colored.clip(min=0, max=1)
        save_video(depth_colored[:num_origin_views], os.path.join(save_path, f"predict_context_depth.mp4"))
        save_video(video[:num_origin_views], os.path.join(save_path, f"predict_context_rgb.mp4"))
        if num_views > num_origin_views:
            save_video(depth_colored[num_origin_views:], os.path.join(save_path, f"predict_interpolate_depth.mp4"))
            save_video(video[num_origin_views:], os.path.join(save_path, f"predict_interpolate_context_rgb.mp4"))

  
    
    def update_batch_by_diffusion(self, batch: BatchedExample) -> BatchedExample:
        """Update the batch by diffusion."""
        if self.model._gaussians is None or self.last_batch_context_extrinsics is None or self.last_encoder_output_extrisics is None:
            print("No gaussians in the model, skipping diffusion.")
            return batch, None
        # extra_extrinsics = extrapolate_extrinsics(batch["context"]["extrinsics"], 5, num_exploration_views)
        # extra_intrinsics = batch["context"]["intrinsics"][-1,-1].repeat(1, num_exploration_views, 1, 1)  
        interpolate_num = 1
        # extra_extrinsics, extra_intrinsics = interpolate_trajectory(
        #     batch["context"]["extrinsics"], batch["context"]["intrinsics"], interpolate_num
        # )

        extra_intrinsics = batch["target"]["intrinsics"]
        with torch.no_grad():
            extra_extrinsics = align_and_transform(self.last_batch_context_extrinsics,
                                                   self.last_encoder_output_extrisics,
                                                   batch["target"]["extrinsics"])
        b, v, c, h, w = batch["context"]["image"].shape
        extra_b, extra_v, _ , _ = extra_extrinsics.shape
        render_result = self.model.get_gaussian_splat_results(extra_extrinsics, extra_intrinsics, h, w, v)
        if render_result is None:
            print("No render result, skipping batch.")
            return batch, None    
        
        extended_batch = copy.deepcopy(batch) 
        diffix_images = self._diffix_util.process_images(
            input_images=render_result.color,
            ref_image=batch["context"]["image"])

        extended_batch["context"]["image"] = F.interpolate(
            diffix_images.reshape(extra_b * extra_v, c, diffix_images.shape[-2] , diffix_images.shape[-1]),
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        ).reshape(extra_b, extra_v, c, h, w) 

        print("render_result.color shape: ", render_result.color.shape, \
              "ref_image shape: ", batch["context"]["image"].shape, \
              "extended_batch image shape: ", extended_batch["context"]["image"].shape)

        extended_batch["context"]["depth"] = render_result.depth
        extend_indexes = torch.arange(extra_v, device=batch["context"]["image"].device) + batch["context"]["index"][0][-1] + 1
        extend_indexes = extend_indexes.unsqueeze(0)
        extended_batch["context"]["index"] =  extend_indexes
        extended_batch["context"]["extrinsics"] = extra_extrinsics
        extended_batch["context"]["intrinsics"] = extra_intrinsics


        full_batch = copy.deepcopy(batch) 
        full_batch["context"]["image"] = torch.cat([batch["context"]["image"], extended_batch["context"]["image"]], dim=1)
        full_batch["context"]["depth"] = torch.cat([batch["context"]["depth"], render_result.depth], dim=1)
        extend_indexes = torch.arange(extra_v, device=batch["context"]["image"].device) + batch["context"]["index"][0][-1] + 1
        extend_indexes = extend_indexes.unsqueeze(0)
        full_batch["context"]["index"] = torch.cat([batch["context"]["index"], extend_indexes], dim=1)
        full_batch["context"]["extrinsics"] = torch.cat([batch["context"]["extrinsics"], extra_extrinsics], dim=1)
        full_batch["context"]["intrinsics"] = torch.cat([batch["context"]["intrinsics"], extra_intrinsics], dim=1)

        if self.global_step % self.detailed_save_interval == 0:
            global_step_save_folder = str(self.train_cfg.output_path / f"steps_{self.global_step}_train_log")
            if not os.path.exists(global_step_save_folder):
                os.makedirs(global_step_save_folder)
            origin_image =  (batch["context"]["image"]+ 1) / 2
            save_video(origin_image[0], os.path.join(global_step_save_folder, f"origin_rgb.mp4"))
            save_video(render_result.color[0], os.path.join(global_step_save_folder, f"rendered_rgb.mp4"))
            save_video(extended_batch["context"]["image"][0], os.path.join(global_step_save_folder, f"diffix_rgb.mp4"))
        
        return full_batch, extended_batch


    def has_sufficient_space(self, path, min_space_gb=10):
        usage = shutil.disk_usage(path)
        print(f"Disk space - Total: {usage.total // (1024**3)} GB, Used: {usage.used // (1024**3)} GB, Free: {usage.free // (1024**3)} GB")
        return usage.free > min_space_gb * (1024**3)

    def split_predict_result(self, encoder_ouput, output, origin_num):
        origin_encoder_output = EncoderOutput(
                        gaussians=encoder_ouput.gaussians,
                        pred_pose_enc_list=[pred_pose_enc[:,:origin_num, :] for pred_pose_enc in encoder_ouput.pred_pose_enc_list],
                        pred_context_pose=dict(
                            extrinsic=encoder_ouput.pred_context_pose['extrinsic'][:,:origin_num],
                            intrinsic=encoder_ouput.pred_context_pose['intrinsic'][:,:origin_num],
                        ),
                        depth_dict=dict(depth=encoder_ouput.depth_dict['depth'][:,:origin_num], 
                                        conf_valid_mask=encoder_ouput.depth_dict['conf_valid_mask'][:,:origin_num]),
                        infos= encoder_ouput.infos,
                        distill_infos=dict( 
                                    pred_pose_enc_list=[pred_pose_enc[:,:origin_num, :] for pred_pose_enc in encoder_ouput.distill_infos['pred_pose_enc_list']] ,
                                    pts_all=encoder_ouput.distill_infos['pts_all'][:,:origin_num], 
                                    depth_map=encoder_ouput.distill_infos['depth_map'][:,:origin_num],
                                    conf_mask=encoder_ouput.distill_infos['conf_mask'][:,:origin_num]
                                    ) if 'pred_pose_enc_list' in encoder_ouput.distill_infos else {},
                        )
        extended_encoder_output = EncoderOutput(
                        gaussians=encoder_ouput.gaussians,
                        pred_pose_enc_list=[pred_pose_enc[:,origin_num:, :] for pred_pose_enc in encoder_ouput.pred_pose_enc_list],
                        pred_context_pose=dict(
                            extrinsic=encoder_ouput.pred_context_pose['extrinsic'][:,origin_num:],
                            intrinsic=encoder_ouput.pred_context_pose['intrinsic'][:,origin_num:],
                        ),
                        depth_dict=dict(depth=encoder_ouput.depth_dict['depth'][:,origin_num:], 
                                        conf_valid_mask=encoder_ouput.depth_dict['conf_valid_mask'][:,origin_num:]),
                        infos= encoder_ouput.infos,
                        distill_infos=dict( 
                                    pred_pose_enc_list=[pred_pose_enc[:,origin_num:, :] for pred_pose_enc in encoder_ouput.distill_infos['pred_pose_enc_list']] ,
                                    pts_all=encoder_ouput.distill_infos['pts_all'][:,origin_num:], 
                                    depth_map=encoder_ouput.distill_infos['depth_map'][:,origin_num:],
                                    conf_mask=encoder_ouput.distill_infos['conf_mask'][:,origin_num:]
                                    ) if 'pred_pose_enc_list' in encoder_ouput.distill_infos else {},
                        )


        
        origin_output = DecoderOutput(
                         color=output.color[:,:origin_num,],
                         depth=output.depth[:,:origin_num,],
                         alpha=output.alpha[:,:origin_num,],
                         lod_rendering = output.lod_rendering
                        )
        
        extend_output = DecoderOutput(
                         color=output.color[:,origin_num:,],
                         depth=output.depth[:,origin_num:,],
                         alpha=output.alpha[:,origin_num:,],
                         lod_rendering = output.lod_rendering
                        )

        
        return origin_encoder_output, extended_encoder_output, origin_output, extend_output
    
    def compute_metrics(self, batch, encoder_output, output):
        target_gt = (batch["context"]["image"]+ 1) / 2
        depth_dict = encoder_output.depth_dict
        distill_infos = encoder_output.distill_infos

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())
        if self.model.encoder.distill:
            consis_absrel = abs_relative_difference(
                rearrange(output.depth, "b v h w -> (b v) h w"),
                rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
                rearrange(distill_infos['conf_mask'], "b v h w -> (b v) h w"),
            )
            self.log("train/consis_absrel", consis_absrel.mean())
            consis_delta1 = delta1_acc(
                rearrange(output.depth, "b v h w -> (b v) h w"),
                rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
                rearrange(distill_infos['conf_mask'], "b v h w -> (b v) h w"),
            )
            self.log("train/consis_delta1", consis_delta1.mean())

    def compute_loss(self, batch, extended_batch, encoder_output, output):
           # Compute and log loss.
        total_loss = 0
        gaussians, pred_pose_enc_list, depth_dict = encoder_output.gaussians, encoder_output.pred_pose_enc_list, encoder_output.depth_dict
        distill_infos = encoder_output.distill_infos
        depth_dict['distill_infos'] = distill_infos
        if self.global_step % self.log_save_interval == 0:
            print("=============global step ", self.global_step, " loss=============")
        b, v, c, h, w = batch["context"]["image"].shape
        with torch.amp.autocast('cuda', enabled=False):
            for loss_fn in self.losses:
                loss = loss_fn.forward(output, batch, gaussians, depth_dict, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                if self.global_step % self.log_save_interval == 0:
                    print(f"loss/{loss_fn.name}:{loss} ")
                total_loss = total_loss + loss
            context_gt_img = (batch["target"]["image"] + 1) / 2
            loss_context_ssim = 1.0 - self.ssim_loss_module(rearrange(output.color, "b v c h w -> (b v) c h w"), 
                                                            rearrange(context_gt_img, "b v c h w -> (b v) c h w"))
           
            total_loss += loss_context_ssim

            if depth_dict is not None and "depth" in get_cfg()["loss"].keys() and self.train_cfg.cxt_depth_weight > 0:
                depth_loss_idx = list(get_cfg()["loss"].keys()).index("depth")
                depth_loss_fn = self.losses[depth_loss_idx].ctx_depth_loss
                loss_depth = depth_loss_fn(depth_dict["depth_map"], depth_dict["depth_conf"], batch, cxt_depth_weight=self.train_cfg.cxt_depth_weight)
                if self.global_step % self.log_save_interval == 0:
                    print("loss/ctx_depth", loss_depth)
                self.log("loss/ctx_depth", loss_depth)
                total_loss = total_loss + loss_depth

            if self.model.encoder.distill:
                # distill ctx pred_pose & depth & normal
                loss_distill_list = self.loss_distill(distill_infos, pred_pose_enc_list, output, batch)
                self.log("loss/distill", loss_distill_list['loss_distill'])
                self.log("loss/distill_pose", loss_distill_list['loss_pose'])
                self.log("loss/distill_depth", loss_distill_list['loss_depth'])
                self.log("loss/distill_normal", loss_distill_list['loss_normal'])
                if self.global_step % self.log_save_interval == 0:
                    print("loss/distill ", loss_distill_list['loss_distill'])
                    print("loss/distill_pose ", loss_distill_list['loss_pose'])
                    print("loss/distill_depth ", loss_distill_list['loss_depth'])
                    print("loss/distill_normal ", loss_distill_list['loss_normal'])
                total_loss = total_loss + loss_distill_list['loss_distill']
            loss_context_pose = self.loss_pose(pred_pose_enc_list, batch, start_index=0, end_index=v)
            total_loss += 0.1 * loss_context_pose["loss_camera"]
            if gaussians is not None:
                with torch.no_grad():
                    target_extrinsics = align_and_transform(batch["context"]["extrinsics"], 
                                                            encoder_output.pred_context_pose['extrinsic'], 
                                                            batch["target"]["extrinsics"])
                render_result = self.model.get_gaussian_splat_results(target_extrinsics,
                                                                      batch["target"]["intrinsics"], 
                                                                      h, w, v)
            
                rendered_rgb = render_result.color
                target_gt_img = (batch["target"]["image"] + 1) / 2
                delta = rendered_rgb - target_gt_img
                loss_prediction_rgb = get_cfg()[ 'loss']['mse']['weight'] * torch.nan_to_num((delta**2).mean(), nan=0.0, posinf=0.0, neginf=0.0)
                lpips_loss_idx = list(get_cfg()["loss"].keys()).index("lpips")
                loss_prediction_lpips =  torch.nan_to_num(self.losses[lpips_loss_idx].lpips.forward(rearrange(rendered_rgb, "b v c h w -> (b v) c h w"), 
                                                                                                    rearrange(target_gt_img, "b v c h w -> (b v) c h w"), 
                                                                                                    normalize=True).mean(), 
                                                                                                    nan=0.0, posinf=0.0, neginf=0.0)

                loss_target_ssim = 1 -  self.ssim_loss_module(rearrange(rendered_rgb, "b v c h w -> (b v) c h w"), 
                                                              rearrange(target_gt_img, "b v c h w -> (b v) c h w"))                                                                            
                total_loss += loss_prediction_rgb + loss_prediction_lpips + loss_target_ssim
                global_step_save_folder = str(self.train_cfg.output_path / f"steps_{self.global_step}_train_log")
                self.log("loss/loss_prediction_rgb", loss_prediction_rgb)
                if self.global_step % self.detailed_save_interval == 0:
                    save_video(target_gt_img[0], os.path.join(global_step_save_folder, f"nvs_gt_image.mp4"))
                    save_video(rendered_rgb[0], os.path.join(global_step_save_folder, f"nvs_render_image.mp4"))
                    ground_truth_pose_file = os.path.join(global_step_save_folder, "ground_truth_poses.pkl")
                    save_poses(ground_truth_pose_file, batch["context"]["extrinsics"], batch["target"]["extrinsics"])
                    predict_pose_file = os.path.join(global_step_save_folder, "predict_poses.pkl")
                    save_poses(predict_pose_file, encoder_output.pred_context_pose['extrinsic'], target_extrinsics)
            if extended_batch is not None:
                diffusion_image = (extended_batch["context"]["image"] + 1) / 2
                target_gt_img = (batch["target"]["image"] + 1) / 2
                lpips_loss_idx = list(get_cfg()["loss"].keys()).index("lpips")
                delta = diffusion_image - target_gt_img
                loss_diffusion_rgb = get_cfg()[ 'loss']['mse']['weight'] * torch.nan_to_num((delta**2).mean(), nan=0.0, posinf=0.0, neginf=0.0)
                loss_diffusion_lpips =  self.losses[lpips_loss_idx].lpips.forward(rearrange(target_gt_img, "b v c h w -> (b v) c h w"), 
                                                                                  rearrange(diffusion_image, "b v c h w -> (b v) c h w"),         
                                                                                  normalize=True).mean()
                loss_diffusion_ssim = 1.0 - self.ssim_loss_module(rearrange(target_gt_img, "b v c h w -> (b v) c h w"),
                                                                  rearrange(diffusion_image, "b v c h w -> (b v) c h w"))
                total_loss += loss_diffusion_rgb + loss_diffusion_lpips + loss_diffusion_ssim

        if self.global_step % self.log_save_interval == 0:                                                    
            print(f"loss/total: {total_loss} ")
            print(f"loss/context_ssim: {loss_context_ssim}", )
            print(f"loss/prediction_rgb: {loss_prediction_rgb}")
            print(f"loss/context_pose: {loss_context_pose}")
            print(f"loss/target_ssim: {loss_target_ssim}")
            print(f"loss/prediction_lpips: {loss_prediction_lpips}")
            if extended_batch is not None:
                print(f"loss/diffusion_rgb: {loss_diffusion_rgb}")
                print(f"loss/diffusion_lpips: {loss_diffusion_lpips}")
                print(f"loss/diffusion_ssim: {loss_diffusion_ssim}")

        return total_loss

    def save_for_visualization(self, num_views, num_origin_views, output, encoder_output):
        global_step_save_folder = str(self.train_cfg.output_path / f"steps_{self.global_step}_train_log")
        if not os.path.exists(global_step_save_folder):
            os.makedirs(global_step_save_folder)
        plyfile = os.path.join(global_step_save_folder, "gaussians.ply")     
        print(f"Exporting Gaussians to {plyfile}")
        gaussians = encoder_output.gaussians
        export_ply(
            gaussians.means[0],
            gaussians.scales[0],
            gaussians.rotations[0],
            gaussians.harmonics[0],
            gaussians.opacities[0],
            Path(plyfile),
            save_sh_dc_only=True,
        )
        self.save_depth_and_rgb(num_views, num_origin_views, output.color[0].clip(min=0, max=1), output.depth[0], global_step_save_folder)
        pred_all_extrinsic = encoder_output.pred_context_pose['extrinsic']
        predict_context_extrinsics = pred_all_extrinsic[:, :num_origin_views]
        predict_extra_extrinsics = pred_all_extrinsic[:, num_origin_views:]
        save_predict_poses_file = os.path.join(global_step_save_folder, "predict_poses.pkl")
        save_poses(save_predict_poses_file, predict_context_extrinsics, predict_extra_extrinsics)

    def print_gpu_memory(self, prefix, unit="GB", rank=0):
        """
        打印当前进程可见的 GPU 显存占用。
        参数
        ----
        unit : str, 可选 "B", "KB", "MB", "GB"
        rank : int, 要查看的卡号；默认 0。若机器只有 1 张卡可忽略。
        """
        # 换算因子
        div = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit]

        # 确保该卡对当前进程可见
        if rank >= torch.cuda.device_count():
            print(f"GPU {rank} 不存在，当前机器只有 {torch.cuda.device_count()} 张卡")
            return

        with torch.cuda.device(rank):
            # 1. PyTorch 已分配（张量实际占用）
            allocated = torch.cuda.memory_allocated()          # 字节
            # 2. PyTorch 已预留（缓存池）
            reserved  = torch.cuda.memory_reserved()           # 字节
            # 3. 驱动级总占用 / 总容量
            free, total = torch.cuda.mem_get_info()            # 字节

        print(f"==={prefix} GPU {rank} 显存快照 ({unit}) ===")
        print(f"PyTorch Allocated : {allocated/div:7.3f} {unit}")
        print(f"PyTorch Reserved  : {reserved /div:7.3f} {unit}")
        print(f"Driver Used       : {(total-free)/div:7.3f} {unit}")
        print(f"Driver Total      : {total/div:7.3f} {unit}")
        print("-" * 35)


    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        # torch.cuda.empty_cache()
        # if self.has_sufficient_space(str(self.train_cfg.output_path)) == False:
        #     raise RuntimeError("Not enough disk space, stopping training to avoid OOM.")
        
        self.model.encoder.update_attention(batch["context"]["patch_width"], batch["context"]["patch_height"])
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined
        # self.print_gpu_memory("Step1")
        batch: BatchedExample = self.data_shim(batch)
        b, v, c, h, w = batch["context"]["image"].shape
       
        if v >= 4 and self.global_step >= 1000:
            full_batch, extended_batch = self.update_batch_by_diffusion(batch)
            self.is_update_by_diffusion = True
        else:
            full_batch = batch
            extended_batch = None
            self.is_update_by_diffusion = False
        context_image = (full_batch["context"]["image"] + 1) / 2
        # Run the model.
        visualization_dump = None
        encoder_output, extended_output = self.model(context_image, self.global_step, visualization_dump=visualization_dump)
        origin_encoder_output, extended_encoder_output, origin_output, extend_output = self.split_predict_result(encoder_output, extended_output, v)
        # self.print_gpu_memory("Step2")
        pred_context_pose = encoder_output.pred_context_pose
        infos = encoder_output.infos
        scene_scale = infos["scene_scale"]
        self.log("train/scene_scale", infos["scene_scale"])
        self.log("train/voxelize_ratio", infos["voxelize_ratio"])
        using_index = torch.arange(v, device=encoder_output.gaussians.means.device)
        batch["using_index"] = using_index
        total_loss = self.compute_loss(batch, extended_batch, origin_encoder_output, origin_output)
        if self.global_step % self.detailed_save_interval == 0:
            self.compute_metrics(batch, origin_encoder_output, origin_output)
            self.save_for_visualization(full_batch["context"]["image"].shape[1], batch["context"]["image"].shape[1], origin_output, origin_encoder_output)

        # self.print_gpu_memory("Step3")
        # Skip batch if loss is too high after certain step

        SKIP_AFTER_STEP = 2000  
        LOSS_THRESHOLD = 5.0
        pred_all_extrinsic = pred_context_pose['extrinsic']
        if self.global_step > SKIP_AFTER_STEP and total_loss > LOSS_THRESHOLD:
            print(f"Skipping batch with high loss ({total_loss:.6f}) at step {self.global_step} on Rank {self.global_rank}")
            # set to a really small number
            return total_loss * 1e-10

        if (
            self.global_rank == 0
            and (self.global_step % self.train_cfg.print_log_every_n_steps == 0)
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}; "
            )

        self.log("info/global_step", self.global_step)  # hack for ckpt monitor
        # self.print_gpu_memory("Step4")
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        self.last_batch_context_extrinsics = copy.deepcopy(batch["context"]["extrinsics"].detach())
        self.last_encoder_output_extrisics = copy.deepcopy(pred_all_extrinsic.detach())

        del batch
        if self.global_step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss
    
    def on_after_backward(self):
        total_norm = 0.0
        counter = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                counter += 1
        total_norm = (total_norm / counter) ** 0.5
        self.log("loss/grad_norm", total_norm)
        
    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")
        
        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.model.encoder(
                (batch["context"]["image"]+1)/2,
                self.global_step,
            )[0]
        # export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path("gaussians.ply"))
        # align the target pose
        if self.test_cfg.align_pose:
            output = self.test_step_align(batch, gaussians)
        else:
            with self.benchmarker.time("decoder", num_calls=v):
                output = self.model.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )
        
        # compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            all_metrics = {
                f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
            }
            methods = ['ours']

            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)
        
        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], output.color[0]):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in output.color[0]],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0])
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.png")
                
    def test_step_align(self, batch, gaussians):
        self.model.encoder.eval()
        # freeze all parameters
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        output_c2ws = batch["target"]["extrinsics"]
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 6], requires_grad=True, device=output_c2ws.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=output_c2ws.device))
            opt_params = []
            self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).to(output_c2ws))
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": 0.005,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": 0.005,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)
            extrinsics = output_c2ws.clone()
            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()
                    dx, drot = cam_trans_delta, cam_rot_delta
                    rot = rotation_6d_to_matrix(
                        drot + self.identity.expand(b, v, -1)
                    )  # (..., 3, 3)

                    transform = torch.eye(4, device=extrinsics.device).repeat((b, v, 1, 1))
                    transform[..., :3, :3] = rot
                    transform[..., :3, 3] = dx

                    new_extrinsics = torch.matmul(extrinsics, transform)
                    output = self.model.decoder.forward(
                        gaussians,
                        new_extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        # cam_rot_delta=cam_rot_delta,
                        # cam_trans_delta=cam_trans_delta,
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    pose_optimizer.step()
                    
        # Render Gaussians.
        output = self.model.decoder.forward(
            gaussians,
            new_extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        return output

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):        
        batch: BatchedExample = self.data_shim(batch)
        print(f"Validation step {self.global_step} on rank {self.global_rank}, batch_idx {batch_idx}, dataloader_idx {dataloader_idx}.")
        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        visualization_dump = {}
        print("model infer start in validation step")
        encoder_output, output = self.model(batch["context"]["image"], self.global_step, visualization_dump=visualization_dump)
        print("model infer finished in validation step")

        gaussians, pred_pose_enc_list, depth_dict = encoder_output.gaussians, encoder_output.pred_pose_enc_list, encoder_output.depth_dict
        pred_context_pose, distill_infos = encoder_output.pred_context_pose, encoder_output.distill_infos
        infos = encoder_output.infos

        GS_num = infos['voxelize_ratio'] * (h*w*v)
        self.log("val/GS_num", GS_num)
        
        num_context_views = pred_context_pose['extrinsic'].shape[1]
        num_target_views = batch["target"]["extrinsics"].shape[1]
        rgb_pred = output.color[0].float()
        depth_pred = vis_depth_map(output.depth[0])

        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        print("Validation rendering finished, computing metrics...")

        # Compute validation metrics.
        rgb_gt = (batch["context"]["image"][0].float() + 1) / 2
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)
        print("Validation metrics computed." )
        # depth metrics
        consis_absrel = abs_relative_difference(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
        )
        self.log("val/consis_absrel", consis_absrel.mean())
        
        consis_delta1 = delta1_acc(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
            valid_mask=rearrange(torch.ones_like(output.depth, device=output.depth.device, dtype=torch.bool), "b v h w -> (b v) h w"),
        )
        self.log("val/consis_delta1", consis_delta1.mean())

        diff_map = torch.abs(output.depth - depth_dict['depth'].squeeze(-1))
        if distill_infos:
            self.log("val/consis_mse", diff_map[distill_infos['conf_mask']].mean())

        # Construct comparison image.
        context_img = inverse_normalize(batch["context"]["image"][0])
        # context_img_depth = vis_depth_map(gaussian_means)
        context = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            # context.append(context_img_depth[i])
        
        colored_diff_map = vis_depth_map(diff_map[0], near=torch.tensor(1e-4, device=diff_map.device), far=torch.tensor(1.0, device=diff_map.device))
        model_depth_pred = depth_dict["depth"].squeeze(-1)[0]
        model_depth_pred = vis_depth_map(model_depth_pred)
        
        render_normal = (get_normal_map(output.depth.flatten(0, 1), batch["context"]["intrinsics"].flatten(0, 1)).permute(0, 3, 1, 2) + 1) / 2.
        pred_normal = (get_normal_map(depth_dict['depth'].flatten(0, 1).squeeze(-1), batch["context"]["intrinsics"].flatten(0, 1)).permute(0, 3, 1, 2) + 1) / 2.

        comparison = hcat(
            add_label(vcat(*context), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
            add_label(vcat(*model_depth_pred), "Depth (VGGT Prediction)"),
            add_label(vcat(*render_normal), "Normal (Prediction)"),
            add_label(vcat(*pred_normal), "Normal (VGGT Prediction)"),
            add_label(vcat(*colored_diff_map), "Diff Map"),
        )

        comparison = torch.nn.functional.interpolate(
            comparison.unsqueeze(0), 
            scale_factor=0.5, 
            mode='bicubic', 
            align_corners=False
        ).squeeze(0)
        
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # self.logger.log_image(
        #     key="comparison",
        #     images=[wandb.Image(prep_image(add_border(comparison)), caption=batch["scene"], file_type="jpg")],
        #     step=self.global_step
        # )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.

        # if isinstance(gaussians, Gaussians):
        #     projections = hcat(
        #             *render_projections(
        #                 gaussians,
        #                 256,
        #                 extra_label="",
        #             )[0]
        #         )
        #     self.logger.log_image(
        #         "projection",
        #         [prep_image(add_border(projections))],
        #         step=self.global_step,
        #     )

        # Draw cameras.
        # cameras = hcat(*render_cameras(batch, 256))
        # self.logger.log_image(
        #     "cameras", [prep_image(add_border(cameras))], step=self.global_step
        # )
        print("Validation images logged." )
        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)
        
        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        encoder_output = self.model.encoder((batch["context"]["image"]+1)/2, self.global_step)
        gaussians, pred_pose_enc_list = encoder_output.gaussians, encoder_output.pred_pose_enc_list

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.model.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }
            
        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=30)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if "gaussian_param_head" in name or "interm" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)
        
        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
