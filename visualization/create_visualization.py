import os
import subprocess

import cv2
import torch
import numpy as np
import pandas as pd
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    MeshRendererWithFragments,
    PointLights,
)
from pytorch3d.transforms import RotateAxisAngle

from IPython import embed; from sys import exit

def create_viz(out_fpath, verts, Rs, Ss, Ts, means, stds, faces, bg, device='cuda:0', comment={}):

    """
    Args:
        verts (torch.Tensor): hand mesh vertices (N, 778, 3)
        Rs (torch.Tensor): rotation matrices (N, 3, 3)
        Ss (torch.Tensor): scale parameter (N)
        Ts (torch.Tensor): translation vectors (N, 2)
        means (torch.Tensor): x and y means (2,)
        stds (torch.Tensor): x and y standard deviations (2,)
        device (str): device to be used during computation

    Original camera parameters:
        Image Center:
            u0= 315.944855;
            v0= 245.287079;
        Focal Lengths:
            fx = 475.065948;
            fy = 475.065857;
    """

    device = torch.device(device)
    N = verts.shape[0]
    img_size = bg[0].shape[:2]

    # Create intrinsic camera matrix (orthographic) (N, 4, 4)
    K = torch.eye(4).float()
    K[0,0] = -1. # Negative focal length
    K[1,1] = -1. # Negative focal length
    K[2,2] = 1.
    K = K.unsqueeze(0).repeat((N, 1, 1))

    # Create rotation matrix (N, 3, 3)
    R = Rs * Ss[..., None, None]

    # Create translation vector
    T = torch.ones(N, 3).float()
    T[:, :2] = Ts

    # Add mean and standard deviations
    R[:, :2] *= stds[None, :, None]
    T[:, :2] *= stds[None, :]
    T[:, :2] += means[None, :]
    
    # Create renderer
    textures = TexturesVertex(verts_features=torch.ones_like(verts).to(device))
    faces = faces.unsqueeze(0).repeat(N, 1, 1)
    
    mesh = Meshes(
        verts=verts.to(device),
        faces=faces.to(device),
        textures=textures
    )
    
    image_size = torch.tensor(img_size).unsqueeze(0).repeat(N, 1)
    cameras = PerspectiveCameras(
        R=R.mT.to(device),
        T=T.to(device),
        K=K.to(device),
        device=device,
        in_ndc=False,
        image_size=image_size,
    )
    # Create mesh rasterizer 
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    # Set lights and shader
    lights = PointLights(device=device, location=[[-1., -1., -1.]])
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    # Do rendering
    renderer = MeshRendererWithFragments(rasterizer, shader)
    images, fragments = renderer(mesh)

    # Alpha blending; background and hand mesh images
    images = (images[...,:3] * 255.).detach().cpu().numpy()
    bg = np.stack(bg)
    alpha_a = (fragments.pix_to_face != -1).squeeze(0).float().cpu().numpy()
    alpha_a *= 1.
    alpha_b = 1.
    alpha_o = alpha_a + alpha_b * (1 - alpha_a)
    img_o = (images * alpha_a + bg * alpha_b * (1 - alpha_a)) / alpha_o

    # Create video
    frame_rate = 30.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_fpath, fourcc, frame_rate, img_size[::-1])
    for f_idx, f in enumerate(img_o):
        cv2.putText(f, f'frame: {f_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(f.astype(np.uint8))
    out.release()

    if comment:
        comment_str = "#".join([f"{k}={v}"for k, v in comment.items()])
        tmp_fpath = os.path.join(os.path.split(out_fpath)[0], "tmp.mp4")
        cmd = [
            "ffmpeg",
            "-i",
            out_fpath,
            "-metadata",
            f"comment={comment_str}",
            "-codec",
            "copy",
            tmp_fpath
        ]
        subprocess.run(cmd)
        os.remove(out_fpath)
        os.rename(tmp_fpath, out_fpath)

if __name__ == "__main__":
    MAX_FRAMES = 250
    out_fpath = "./video_id_1.mp4" 
    c_r =  torch.from_numpy(np.load("./R_video_id_1.npy"))
    c_s =  torch.from_numpy(np.load("./S_video_id_1.npy"))
    c_o =  torch.from_numpy(np.load("./O_video_id_1.npy"))
    hand_meshes =  torch.from_numpy(np.load("./3D_hand_meshes_video_id_1.npy"))
    means = torch.from_numpy(np.load('/home/tmpvideos/SLR/HANDS17/preprocess/means.npy')).squeeze()
    stds = torch.from_numpy(np.load('/home/tmpvideos/SLR/HANDS17/preprocess/stds.npy')).squeeze()
    faces = torch.from_numpy(np.load("./MANO_faces.npy").astype(np.int32))
    # Grab original frames test split
    data_df = pd.read_csv('/home/tmpvideos/SLR/HANDS17/preprocess/test.csv')
    seq_or_df = data_df[data_df.iloc[:,0].str.contains(f'tracking\\1\\images', regex=False)]
    seq_or_imgs_fpaths = seq_or_df.iloc[:,0].values.tolist()
    seq_or_imgs_fpaths = [os.path.join('/home/tmpvideos/SLR/HANDS17/', fpath) for fpath in seq_or_imgs_fpaths]
    seq_or_imgs_fpaths = [fpath.replace('\\', '/') for fpath in seq_or_imgs_fpaths]
    tracking_imgs = [cv2.imread(fpath, cv2.IMREAD_UNCHANGED) for fpath in seq_or_imgs_fpaths]
    tracking_imgs = [cv2.normalize(ti, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) for ti in tracking_imgs]
    tracking_imgs = [cv2.applyColorMap(ti, cv2.COLORMAP_BONE) for ti in tracking_imgs]

    hand_meshes = hand_meshes[:MAX_FRAMES]
    c_r = c_r[:MAX_FRAMES]
    c_s = c_s[:MAX_FRAMES]
    c_o = c_o[:MAX_FRAMES]
    tracking_imgs = tracking_imgs[:MAX_FRAMES]
    
    create_viz(
        out_fpath,
        verts=hand_meshes,
        Rs=c_r,
        Ss=c_s,
        Ts=c_o,
        means=means,
        stds=stds,
        faces=faces,
        bg=tracking_imgs,
        device="cpu"
    )