from pathlib import Path
import os
import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor
import pickle



def save_poses(save_path: str,
                       pred_all_context_extrinsic: torch.Tensor,
                       pred_all_target_extrinsic: torch.Tensor):
    """
    一次性保存并可视化两组位姿序列
    :param save_path: 保存文件路径（例如 'poses.pkl'）
    :param pred_all_context_extrinsic: (1, m, 4, 4) tensor
    :param pred_all_target_extrinsic:  (1, n, 4, 4) tensor
    :param axis_len:   朝向箭头长度
    :param stride:     每 stride 帧画一次朝向
    """
    # ---------- 1. 保存 ----------
    context_np = pred_all_context_extrinsic.squeeze(0).cpu().detach().numpy()
    target_np  = pred_all_target_extrinsic.squeeze(0).cpu().detach().numpy()
    m, n = context_np.shape[0], target_np.shape[0]
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({'context': context_np, 'target': target_np, 'm': m, 'n': n}, f)
    print(f'已保存到 {save_path}  (m={m}, n={n})')


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
):
    if shift_and_scale:
        # Shift the scene so that the median Gaussian is at the origin.
        means = means - means.median(dim=0).values

        # Rescale the scene so that most Gaussians are within range [-1, 1].
        scale_factor = means.abs().quantile(0.95, dim=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0 if save_sh_dc_only else f_rest.shape[1])]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    ]
    if save_sh_dc_only:
        # remove f_rest from attributes
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
