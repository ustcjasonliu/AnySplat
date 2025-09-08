import torch
import numpy as np
from einops import einsum, rearrange, reduce
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from typing import Tuple

import roma

def interpolate_intrinsics(
    initial: Float[Tensor, "*#batch 3 3"],
    final: Float[Tensor, "*#batch 3 3"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 3 3"]:
    initial = rearrange(initial, "... i j -> ... () i j")
    final = rearrange(final, "... i j -> ... () i j")
    t = rearrange(t, "t -> t () ()")
    return initial + (final - initial) * t



def intersect_rays(
    a_origins: Float[Tensor, "*#batch dim"],
    a_directions: Float[Tensor, "*#batch dim"],
    b_origins: Float[Tensor, "*#batch dim"],
    b_directions: Float[Tensor, "*#batch dim"],
) -> Float[Tensor, "*batch dim"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast and stack the tensors.
    a_origins, a_directions, b_origins, b_directions = torch.broadcast_tensors(
        a_origins, a_directions, b_origins, b_directions
    )
    origins = torch.stack((a_origins, b_origins), dim=-2)
    directions = torch.stack((a_directions, b_directions), dim=-2)

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "... n i, ... n j -> ... n i j")
    n = n - torch.eye(3, dtype=origins.dtype, device=origins.device)

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "... n i j -> ... i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "... n i j, ... n j -> ... n i")
    rhs = reduce(rhs, "... n i -> ... i", "sum")

    # Left-matrix-multiply both sides by the inverse of lhs to find p.
    return torch.linalg.lstsq(lhs, rhs).solution


def normalize(a: Float[Tensor, "*#batch dim"]) -> Float[Tensor, "*#batch dim"]:
    return a / a.norm(dim=-1, keepdim=True)


def generate_coordinate_frame(
    y: Float[Tensor, "*#batch 3"],
    z: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 3 3"]:
    """Generate a coordinate frame given perpendicular, unit-length Y and Z vectors."""
    y, z = torch.broadcast_tensors(y, z)
    return torch.stack([y.cross(z), y, z], dim=-1)


def generate_rotation_coordinate_frame(
    a: Float[Tensor, "*#batch 3"],
    b: Float[Tensor, "*#batch 3"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch 3 3"]:
    """Generate a coordinate frame where the Y direction is normal to the plane defined
    by unit vectors a and b. The other axes are arbitrary."""
    device = a.device

    # Replace every entry in b that's parallel to the corresponding entry in a with an
    # arbitrary vector.
    b = b.detach().clone()
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 0, 1], dtype=b.dtype, device=device)
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 1, 0], dtype=b.dtype, device=device)
    # Generate the coordinate frame. The initial cross product defines the plane.
    return generate_coordinate_frame(normalize(a.cross(b)), a)


def matrix_to_euler(
    rotations: Float[Tensor, "*batch 3 3"],
    pattern: str,
) -> Float[Tensor, "*batch 3"]:
    *batch, _, _ = rotations.shape
    rotations = rotations.reshape(-1, 3, 3)
    angles_np = R.from_matrix(rotations.detach().cpu().numpy()).as_euler(pattern)
    rotations = torch.tensor(angles_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3)


def euler_to_matrix(
    rotations: Float[Tensor, "*batch 3"],
    pattern: str,
) -> Float[Tensor, "*batch 3 3"]:
    *batch, _ = rotations.shape
    rotations = rotations.reshape(-1, 3)
    matrix_np = R.from_euler(pattern, rotations.detach().cpu().numpy()).as_matrix()
    rotations = torch.tensor(matrix_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3, 3)


def extrinsics_to_pivot_parameters(
    extrinsics: Float[Tensor, "*#batch 4 4"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 5"]:
    """Convert the extrinsics to a representation with 5 degrees of freedom:
    1. Distance from pivot point in the "X" (look cross pivot axis) direction.
    2. Distance from pivot point in the "Y" (pivot axis) direction.
    3. Distance from pivot point in the Z (look) direction
    4. Angle in plane
    5. Twist (rotation not in plane)
    """

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    # Compute the translation elements of the pivot parametrization.
    translation_frame = generate_coordinate_frame(pivot_axis, extrinsics[..., :3, 2])
    origin = extrinsics[..., :3, 3]
    delta = pivot_point - origin
    translation = einsum(translation_frame, delta, "... i j, ... i -> ... j")

    # Add the rotation elements of the pivot parametrization.
    inverted = pivot_coordinate_frame.inverse() @ extrinsics[..., :3, :3]   
    inverted_det = torch.linalg.det(inverted[..., :3, :3])

    extrinsics_det = torch.linalg.det(extrinsics[..., :3, :3])
    pivot_coordinate_frame_det = torch.linalg.det(pivot_coordinate_frame[..., :3, :3])
    pivot_coordinate_frame_inverse_det = torch.linalg.det(pivot_coordinate_frame.inverse()[..., :3, :3])

    if (abs(inverted_det - 1.0) > 1e-4).any():
        raise ValueError("invalid extrinsics, cannot convert to pivot parameters")

    y, _, z = matrix_to_euler(inverted, "YXZ").unbind(dim=-1)

    return torch.cat([translation, y[..., None], z[..., None]], dim=-1)


def pivot_parameters_to_extrinsics(
    parameters: Float[Tensor, "*#batch 5"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 4 4"]:
    translation, y, z = parameters.split((3, 1, 1), dim=-1)

    euler = torch.cat((y, torch.zeros_like(y), z), dim=-1)
    rotation = pivot_coordinate_frame @ euler_to_matrix(euler, "YXZ")

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    translation_frame = generate_coordinate_frame(pivot_axis, rotation[..., :3, 2])
    delta = einsum(translation_frame, translation, "... i j, ... j -> ... i")
    origin = pivot_point - delta

    *batch, _ = origin.shape
    extrinsics = torch.eye(4, dtype=parameters.dtype, device=parameters.device)
    extrinsics = extrinsics.broadcast_to((*batch, 4, 4)).clone()
    extrinsics[..., 3, 3] = 1
    extrinsics[..., :3, :3] = rotation
    extrinsics[..., :3, 3] = origin
    return extrinsics


def interpolate_circular(
    a: Float[Tensor, "*#batch"],
    b: Float[Tensor, "*#batch"],
    t: Float[Tensor, "*#batch"],
) -> Float[Tensor, " *batch"]:
    a, b, t = torch.broadcast_tensors(a, b, t)

    tau = 2 * torch.pi
    a = a % tau
    b = b % tau

    # Consider piecewise edge cases.
    d = (b - a).abs()
    a_left = a - tau
    d_left = (b - a_left).abs()
    a_right = a + tau
    d_right = (b - a_right).abs()
    use_d = (d < d_left) & (d < d_right)
    use_d_left = (d_left < d_right) & (~use_d)
    use_d_right = (~use_d) & (~use_d_left)

    result = a + (b - a) * t
    result[use_d_left] = (a_left + (b - a_left) * t)[use_d_left]
    result[use_d_right] = (a_right + (b - a_right) * t)[use_d_right]

    return result


def interpolate_pivot_parameters(
    initial: Float[Tensor, "*#batch 5"],
    final: Float[Tensor, "*#batch 5"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 5"]:
    initial = rearrange(initial, "... d -> ... () d")
    final = rearrange(final, "... d -> ... () d")
    t = rearrange(t, "t -> t ()")
    ti, ri = initial.split((3, 2), dim=-1)
    tf, rf = final.split((3, 2), dim=-1)

    t_lerp = ti + (tf - ti) * t
    r_lerp = interpolate_circular(ri, rf, t)

    return torch.cat((t_lerp, r_lerp), dim=-1)


@torch.no_grad()
def extrapolate_extrinsics(
    trajectory: torch.Tensor,      # (*, frames, 4, 4)  这里假设 batch=1
    inter_n,
    extrapolate_n: int                        # 需要外推的帧数
) -> torch.Tensor:                # 返回 (n, 4, 4)
    """
    用最近 inter_n 帧（或全部）外推未来 n 帧的 4×4 外参矩阵。
    输入 shape:  (*, frames, 4, 4)  这里 batch 维度被 squeeze 掉
    输出 shape:  (n, 4, 4)
    """
    traj = trajectory.squeeze()          # (frames, 4, 4)
    frames = traj.shape[0]
    k = min(inter_n, frames)                   # 若不足 5 帧就用全部
    recent = traj[-k:].cpu().numpy()           # (k,4,4)
    # 1. 平移向量 t
    t = recent[:, :3, 3]                 # (k, 3)
    # 2. 旋转矩阵 → 四元数 (x,y,z,w)
    rot = R.from_matrix(recent[:, :3, :3])
    q = rot.as_quat()                    # (k,4)
    # 时间轴简单用索引 0..k-1
    t_idx = np.arange(k, dtype=np.float32)
    # 拟合平移：三次多项式
    t_poly = [np.polyfit(t_idx, t[:, i], deg=min(3, k-1)) for i in range(3)]
    # 拟合四元数角速度：用 SLERP 算平均角速度
    if k == 1:
        omega = np.zeros(3)
    else:
        delta_q = (R.from_quat(q[:-1]).inv()
                   * R.from_quat(q[1:]))
        angles = delta_q.magnitude()
        omega = R.from_rotvec(delta_q[0].as_rotvec()
                                     / (t_idx[1] - t_idx[0])).as_rotvec()

    # 外推
    pred_t_list, pred_q_list = [], []
    for i in range(1, extrapolate_n+1):
        new_t = np.array([np.polyval(p, k-1 + i) for p in t_poly])
        pred_t_list.append(new_t)

        step_rot = R.from_rotvec(omega * i)
        new_q = (R.from_quat(q[-1]) * step_rot).as_quat()
        pred_q_list.append(new_q)

    # 组装 4×4
    pred_T = torch.zeros(trajectory.shape[0],extrapolate_n, 4, 4)
    pred_T[:,:, 3, 3] = 1.0
    pred_T[:,:, :3, 3] = torch.tensor(np.stack(pred_t_list))
    pred_R = R.from_quat(np.stack(pred_q_list)).as_matrix()
    pred_T[:, :, :3, :3] = torch.tensor(pred_R)

    return pred_T.cuda()


def linear_interpolate_extrinsics(
    initial: Tensor,  # (n,4,4)
    final: Tensor,    # (n,4,4)
    T_mid: int,       # 要插入的中间帧数
) -> Tensor:          # (n, T_mid, 4, 4)
    n = initial.size(0)
    device = initial.device
    # 生成 (1/(T_mid+1), ..., T_mid/(T_mid+1))
    t = torch.linspace(1/(T_mid+1), T_mid/(T_mid+1), T_mid, device=device)

    R0, t0 = initial[:, :3, :3], initial[:, :3, 3]  # (n,3,3)  (n,3)
    R1, t1 = final[:, :3, :3],   final[:, :3, 3]

    # 旋转 SLERP：roma.rotmat_slerp 返回 (T_mid, n, 3, 3) -> (n, T_mid, 3, 3)
    R_mid = roma.rotmat_slerp(R0, R1, t).permute(1, 0, 2, 3)
    # 平移线性
    t_mid = t0[:, None, :] + t.view(1, T_mid, 1) * (t1[:, None, :] - t0[:, None, :])
    # 拼 4×4
    pose = torch.eye(4, device=device).repeat(n, T_mid, 1, 1)
    pose[:, :, :3, :3] = R_mid
    pose[:, :, :3, 3]  = t_mid
    return pose


@torch.no_grad()
def interpolate_extrinsics(
    initial: Float[Tensor, "*#batch 4 4"],
    final: Float[Tensor, "*#batch 4 4"],
    t: Float[Tensor, " time_step"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch time_step 4 4"]:
    """Interpolate extrinsics by rotating around their "focus point," which is the
    least-squares intersection between the look vectors of the initial and final
    extrinsics.
    """

    initial = initial.type(torch.float64)
    final = final.type(torch.float64)
    t = t.type(torch.float64)

    # Based on the dot product between the look vectors, pick from one of two cases:
    # 1. Look vectors are parallel: interpolate about their origins' midpoint.
    # 3. Look vectors aren't parallel: interpolate about their focus point.
    initial_look = initial[..., :3, 2]
    final_look = final[..., :3, 2]
    dot_products = einsum(initial_look, final_look, "... i, ... i -> ...")
    parallel_mask = (dot_products.abs() - 1).abs() < eps

    # Pick focus points.
    initial_origin = initial[..., :3, 3]
    final_origin = final[..., :3, 3]
    pivot_point = 0.5 * (initial_origin + final_origin)
    pivot_point[~parallel_mask] = intersect_rays(
        initial_origin[~parallel_mask],
        initial_look[~parallel_mask],
        final_origin[~parallel_mask],
        final_look[~parallel_mask],
    )
    
    # Convert to pivot parameters.
    pivot_frame = generate_rotation_coordinate_frame(initial_look, final_look, eps=eps)
    initial_params = extrinsics_to_pivot_parameters(initial, pivot_frame, pivot_point)
    final_params = extrinsics_to_pivot_parameters(final, pivot_frame, pivot_point)

    # Interpolate the pivot parameters.
    interpolated_params = interpolate_pivot_parameters(initial_params, final_params, t)

    # Convert back.
    return pivot_parameters_to_extrinsics(
        interpolated_params.type(torch.float32),
        rearrange(pivot_frame, "... i j -> ... () i j").type(torch.float32),
        rearrange(pivot_point, "... xyz -> ... () xyz").type(torch.float32),
    )



def interpolate_trajectory(extrinsics, intrinsics, inter_n, max_n = 5):
    *B, F, _, _ = extrinsics.shape
    device = extrinsics.device
    dtype = extrinsics.dtype

    # 过滤非法外参
    det = torch.linalg.det(extrinsics[..., :3, :3])
    invalid = abs(det - 1.0) > 1e-4
    if invalid.any():
        print("Warning: filter illegal extrinsics", invalid.sum().item(), "帧")
        extrinsics = extrinsics[..., ~invalid, :, :]
        intrinsics = intrinsics[..., ~invalid, :, :]
        F = extrinsics.size(-3)

    if F < 2:
        raise ValueError("less than 2 valid extrinsics, cannot interpolate")

    t = torch.linspace(0, 1, inter_n + 2, device=device, dtype=torch.float64)[1:-1]

    extr_a = extrinsics[..., :-1, :, :].reshape(-1, 4, 4)
    extr_b = extrinsics[..., 1:,  :, :].reshape(-1, 4, 4)
    intr_a = intrinsics[..., :-1, :, :].reshape(-1, 3, 3)
    intr_b = intrinsics[..., 1:,  :, :].reshape(-1, 3, 3)
    extr_a_det = torch.linalg.det(extr_a[..., :3, :3])
    extr_b_det = torch.linalg.det(extr_b[..., :3, :3])
    if ((abs(extr_a_det - 1.0) > 1e-4).any() or \
        (abs(extr_b_det - 1.0) > 1e-4).any()):
        raise ValueError("invalid extrinsics, cannot interpolate")

    extr_mid = linear_interpolate_extrinsics(extr_a, extr_b, inter_n)
    intr_mid = interpolate_intrinsics(intr_a, intr_b, t)
    # for i in range(extr_b.shape[0]):
    #     print("extr_a", extr_a[i], "extr_b", extr_b[i], "extr_mid", extr_mid[i], "intr_a", intr_a[i], "intr_b", intr_b[i],  "intr_mid", intr_mid[i])
    
    extr_all = extr_mid.reshape(*B, (F - 1) * inter_n, 4, 4).type(dtype)
    intr_all = intr_mid.reshape(*B, (F - 1) * inter_n, 3, 3).type(dtype)
    # if extr_all.size(1) > max_n:
    #     indices = torch.randperm(extr_all.size(1))[:max_n]
    #     extr_all = extr_all[:, indices]
    #     intr_all = intr_all[:, indices]
    #     print(f"Warning: too many interpolated frames, randomly sample {max_n} frames")
    return extr_all, intr_all