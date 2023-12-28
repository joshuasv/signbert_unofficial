import os
import pickle
from shutil import rmtree
import argparse

import cv2
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from signbert.model.SignBertModel import SignBertModel
from signbert.model.SignBertModelManoTorch import SignBertModel as SignBertModelManoTorch

from signbert.data_modules.HANDS17DataModule import HANDS17DataModule

from IPython import embed; from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_fpath", type=str, required=True)
parser.add_argument("--ckpt_fpath", type=str, required=True)
parser.add_argument("--split", choices=["train", "val"], type=str, required=True)
args = parser.parse_args()

# Load config
# config_fpath = "/home/gts/projects/jsoutelo/SignBERT+/configs/msg3d-gesture-extractor-cluster-BMSG3D-bf16.yml"
with open(args.cfg_fpath, 'r') as fid:
    cfg = yaml.load(fid, yaml.SafeLoader)

# Load model
manotorch = cfg.get('manotorch', False)
mano_model_cls = SignBertModelManoTorch if manotorch else SignBertModel
lr = cfg.get('lr', 0.0)
normalize = cfg.get('normalize')
model = mano_model_cls(
    **cfg['model_args'], 
    lr=lr, 
    normalize_inputs=normalize, 
    means_fpath=HANDS17DataModule.MEANS_NPY_FPATH, 
    stds_fpath=HANDS17DataModule.STDS_NPY_FPATH
)

# Load weights
# weights_fpath = '/home/gts/projects/jsoutelo/SignBERT+/logs/msg3d-gesture-extractor-cluster-BMSG3D-bf16/version_0/ckpts/epoch=epoch=3740-step=step=22446-val_PCK_20=0.9065.ckpt'
model = model.load_from_checkpoint(args.ckpt_fpath, map_location='cpu')
model.eval()

# Load data
# split = "train"
hands17 = HANDS17DataModule(batch_size=1, normalize=normalize)
hands17.setup()
if args.split == "train":
    dataloader = hands17.train_dataloader()
elif args.split == "val":
    dataloader = hands17.val_dataloader()
else:
    raise Exception(f"Split {args.split} not available.")
means = np.load(HANDS17DataModule.MEANS_NPY_FPATH)
stds = np.load(HANDS17DataModule.STDS_NPY_FPATH)

# dataloader = hands17.train_dataloader()
# Get random sample
# random_sample = next(iter(dataloader))
# idx, seq_or, seq_in, score, mask_frames_idxs = random_sample
# # Inference
# (seq_out, theta, beta, hand_mesh, c_r, c_s, c_o, center_jt, jt_3d) = model(seq_in)
# # To numpy
# seq_or = seq_or.squeeze(0).numpy()
# seq_in = seq_in.squeeze(0).numpy()
# seq_out = seq_out.squeeze(0).detach().numpy()
# # Remove paddings
# pad_idxs = (seq_or != 0.0).all((1,2))
# seq_or = seq_or[pad_idxs]
# seq_in = seq_in[pad_idxs]
# seq_out = seq_out[pad_idxs]
# Compute PCK@20 for every validation sample
record_idxs = []
record_pck_20_input = []
record_pck_20_output = []
record_auc_pck_20_40_input = []
record_auc_pck_20_40_output = []
record_seq_out = []
record_seq_in = []
record_seq_or = []
record_hand_meshes = []
record_c_r = []
record_c_s = []
record_c_o = []
record_center_jt = []
record_jt_3d = []
record_betas = []
record_thetas = []
for b in dataloader:
    idx, seq_or, seq_in, score, mask_frames_idxs = b
    with torch.no_grad():
        (seq_out, theta, beta, hand_mesh, c_r, c_s, c_o, center_jt, jt_3d) = model(seq_in)
    seq_or = seq_or.squeeze(0).numpy()
    seq_in = seq_in.squeeze(0).numpy()
    seq_out = seq_out.squeeze(0).detach().numpy()
    theta = theta.squeeze(0).detach().numpy()
    beta = beta.squeeze(0).detach().numpy()
    hand_mesh = hand_mesh.squeeze(0).detach().numpy()
    c_r = c_r.squeeze(0).detach().numpy()
    c_s = c_s.squeeze(0).detach().numpy()
    c_o = c_o.squeeze(0).detach().numpy()
    center_jt = center_jt.squeeze(0).detach().numpy()
    jt_3d = jt_3d.squeeze(0).detach().numpy()
    pad_idxs = (seq_or != 0.0).all((1,2))
    seq_or = seq_or[pad_idxs]
    seq_in = seq_in[pad_idxs]
    seq_out = seq_out[pad_idxs]
    beta = beta[pad_idxs]
    theta = theta[pad_idxs]
    hand_mesh = hand_mesh[pad_idxs]
    c_r = c_r[pad_idxs]
    c_s = c_s[pad_idxs]
    c_o = c_o[pad_idxs]
    center_jt = center_jt[pad_idxs]
    jt_3d = jt_3d[pad_idxs]
    # Save data
    record_idxs.append(idx.item())
    record_seq_or.append(seq_or)
    record_seq_in.append(seq_in)
    record_seq_out.append(seq_out)
    record_hand_meshes.append(hand_mesh)
    record_c_r.append(c_r)
    record_c_s.append(c_s)
    record_c_o.append(c_o)
    record_center_jt.append(center_jt)
    record_jt_3d.append(jt_3d)
    record_betas.append(beta)
    record_thetas.append(theta)
    # Compute metrcs
    if normalize:
        seq_or = (seq_or * stds) + means
        seq_in_is_zero = seq_in == 0.0
        seq_in = (seq_in * stds) + means
        seq_in[seq_in_is_zero] = 0.0
        seq_out = (seq_out * stds) + means
    # PCK@20
    input_dists = np.linalg.norm(seq_or - seq_in, axis=-1)
    output_dists = np.linalg.norm(seq_or - seq_out, axis=-1)
    pck_20_input = (input_dists <= 20).sum() / input_dists.size
    pck_20_output = (output_dists <= 20).sum() / output_dists.size
    # AUC PCK@20-40
    input_auc_pck_range_20_40 = torch.tensor([
        (input_dists <= i).sum() / input_dists.size
        for i in range(20, 40)
    ], dtype=torch.float32)
    auc_pck_20_40_input = torch.trapz(input_auc_pck_range_20_40, torch.arange(20, 40)) / (40 - 20)
    output_auc_pck_range_20_40 = torch.tensor([
        (output_dists <= i).sum() / output_dists.size
        for i in range(20, 40)
    ], dtype=torch.float32)
    auc_pck_20_40_output = torch.trapz(output_auc_pck_range_20_40, torch.arange(20, 40)) / (40 - 20)
    
    record_pck_20_input.append(pck_20_input)
    record_pck_20_output.append(pck_20_output)
    record_auc_pck_20_40_input.append(auc_pck_20_40_input)
    record_auc_pck_20_40_output.append(auc_pck_20_40_output)

    print(f"[{idx.item()}] pck_20_input={pck_20_input:.4f} pck_20_output={pck_20_output:.4f} auc_pck_20_40_input={auc_pck_20_40_input:.4f} auc_pck_20_40_output={auc_pck_20_40_output:.4f}")
print(f'[MEAN] pck_20_input={np.mean(record_pck_20_input):.4f} pck_20_output={np.mean(record_pck_20_output):.4f} auc_pck_20_40_input={np.mean(record_auc_pck_20_40_input):.4f} auc_pck_20_40_output={np.mean(record_auc_pck_20_40_output):.4f}')

cfg_name = os.path.split(args.cfg_fpath)[1].replace(".yml", "")
ckpt_name = os.path.split(args.ckpt_fpath)[1].replace(".ckpt", "")
folder_name = os.path.join("./inference_outputs", cfg_name, args.split, ckpt_name)
os.makedirs(f"./{folder_name}", exist_ok=True)
f = open(f'./{folder_name}/record_idxs.pkl', 'wb'); pickle.dump(record_idxs, f); f.close()
f = open(f'./{folder_name}/record_seq_or.pkl', 'wb'); pickle.dump(record_seq_or, f); f.close()
f = open(f'./{folder_name}/record_seq_in.pkl', 'wb'); pickle.dump(record_seq_in, f); f.close()
f = open(f'./{folder_name}/record_seq_out.pkl', 'wb'); pickle.dump(record_seq_out, f); f.close()
f = open(f'./{folder_name}/record_pck_20_input.pkl', 'wb'); pickle.dump(record_pck_20_input, f); f.close()
f = open(f'./{folder_name}/record_pck_20_output.pkl', 'wb'); pickle.dump(record_pck_20_output, f); f.close()
f = open(f'./{folder_name}/record_auc_pck_20_40_input.pkl', 'wb'); pickle.dump(record_auc_pck_20_40_input, f); f.close()
f = open(f'./{folder_name}/record_auc_pck_20_40_output.pkl', 'wb'); pickle.dump(record_auc_pck_20_40_output, f); f.close()
f = open(f'./{folder_name}/record_hand_meshes.pkl', 'wb'); pickle.dump(record_hand_meshes, f); f.close()
f = open(f'./{folder_name}/record_c_r.pkl', 'wb'); pickle.dump(record_c_r, f); f.close()
f = open(f'./{folder_name}/record_c_s.pkl', 'wb'); pickle.dump(record_c_s, f); f.close()
f = open(f'./{folder_name}/record_c_o.pkl', 'wb'); pickle.dump(record_c_o, f); f.close()
f = open(f'./{folder_name}/record_center_jt.pkl', 'wb'); pickle.dump(record_center_jt, f); f.close()
f = open(f'./{folder_name}/record_jt_3d.pkl', 'wb'); pickle.dump(record_jt_3d, f); f.close()
f = open(f'./{folder_name}/record_theta.pkl', 'wb'); pickle.dump(record_thetas, f); f.close()
f = open(f'./{folder_name}/record_beta.pkl', 'wb'); pickle.dump(record_betas, f); f.close()
with open(f'./{folder_name}/metrics.txt', "w") as fid:
    for i, (pck_val_input, pck_val_output, auc_pck_val_input, auc_pck_val_output) in enumerate(zip(record_pck_20_input, record_pck_20_output, record_auc_pck_20_40_input, record_auc_pck_20_40_output)):
        fid.write(f"[{i}] PCK@20_inp={pck_val_input:.4f} PCK@20_out={pck_val_output:.4f} AUC_PCK@20-40_inp={auc_pck_val_input:.4f} AUC_PCK@20-40_out={auc_pck_val_output:.4f}\n")
    fid.write(f"[MEAN] PCK@20_inp={np.mean(record_pck_20_input):.4f} PCK@20_out={np.mean(record_pck_20_output):.4f} AUC_PCK@20-40_inp={np.mean(record_auc_pck_20_40_input):.4f} AUC_PCK@20-40_out={np.mean(record_auc_pck_20_40_output):.4f}\n")
    

seq_idx = 15
frame_idx = 200
np.save(f'./{folder_name}/thetas.npy', record_thetas[seq_idx][frame_idx])
np.save(f'./{folder_name}/betas.npy', record_betas[seq_idx][frame_idx])

# f = open('./record_idxs.pkl', 'rb'); record_idxs = pickle.load(f); f.close()
# f = open('./record_seq_or.pkl', 'rb'); record_seq_or = pickle.load(f); f.close()
# f = open('./record_seq_in.pkl', 'rb'); record_seq_in = pickle.load(f); f.close()
# f = open('./record_seq_out.pkl', 'rb'); record_seq_out = pickle.load(f); f.close()
# f = open('./record_pck_20.pkl', 'rb'); record_pck_20 = pickle.load(f); f.close()
# f = open('./record_hand_meshes.pkl', 'rb'); record_hand_meshes = pickle.load(f); f.close()
# f = open('./record_c_r.pkl', 'rb'); record_c_r = pickle.load(f); f.close()
# f = open('./record_c_s.pkl', 'rb'); record_c_s = pickle.load(f); f.close()
# f = open('./record_c_o.pkl', 'rb'); record_c_o = pickle.load(f); f.close()
# # Grab random sample
# seq_idx = np.random.choice(record_idxs) # From 1 to 99
# seq_or = record_seq_or[seq_idx-1]
# seq_in = record_seq_in[seq_idx-1]
# seq_out = record_seq_out[seq_idx-1]
# seq_pck_20 = record_pck_20[seq_idx-1]
# hand_meshes = record_hand_meshes[seq_idx-1]
# c_r = record_c_r[seq_idx-1]
# c_s = record_c_s[seq_idx-1]
# c_o = record_c_o[seq_idx-1]
# # Grab original frames
# data_df = pd.read_csv('/home/temporal2/jsoutelo/datasets/HANDS17/preprocess/test.csv')
# seq_or_df = data_df[data_df.iloc[:,0].str.contains(f'tracking\\{seq_idx}\\images', regex=False)]
# seq_or_imgs_fpaths = seq_or_df.iloc[:,0].values.tolist()
# seq_or_imgs_fpaths = [os.path.join('/home/temporal2/jsoutelo/datasets/HANDS17', fpath) for fpath in seq_or_imgs_fpaths]
# seq_or_imgs_fpaths = [fpath.replace('\\', '/') for fpath in seq_or_imgs_fpaths]
# tracking_imgs = [cv2.imread(fpath, cv2.IMREAD_UNCHANGED) for fpath in seq_or_imgs_fpaths]
# tracking_imgs = [cv2.normalize(ti, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) for ti in tracking_imgs]
# tracking_imgs = [cv2.applyColorMap(ti, cv2.COLORMAP_BONE) for ti in tracking_imgs]

# from pytorch3d.structures import Meshes
# from pytorch3d.renderer import (
#     look_at_view_transform,
#     FoVPerspectiveCameras,
#     RasterizationSettings,
#     MeshRasterizer,
#     SoftPhongShader,
#     TexturesVertex,
#     MeshRendererWithFragments,
#     PointLights,
#     FoVOrthographicCameras,
# )
# from pytorch3d.common.datatypes import Device

        

# from viz.plot_MANO import MANO_FACES
# device = 'cpu'
# v = torch.tensor(hand_meshes).to(device)
# f = torch.tensor(MANO_FACES.astype(np.int64)).unsqueeze(0).repeat((v.shape[0], 1, 1)).to(device)
# v_rgb = torch.ones_like(v).to(device)
# v_rgb[...,0] = 0.
# v_rgb[...,2] = 0.
# # textures = TexturesVertex(verts_features=v_rgb)
# # mesh = Meshes(
# #     verts=v,
# #     faces=f,
# #     textures=textures
# # )
# # R = torch.tensor(c_r)
# # T = torch.tensor(np.c_[c_o.squeeze(1), np.zeros(c_o.shape[0])])
# textures = TexturesVertex(verts_features=v_rgb[0][None])
# mesh = Meshes(
#     verts=[v[0]],
#     faces=[f[0]],
#     textures=textures
# )
# # R = torch.tensor(c_r)[0][None]
# # T = torch.tensor(np.c_[c_o.squeeze(1), np.zeros(c_o.shape[0])])[0][None]
# R, T = look_at_view_transform(dist=1, elev=0, azim=0)
# # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
# cameras = FoVOrthographicCameras(R=R, T=T, device=device)
# image_size = tracking_imgs[0].shape[:2]
# raster_settings = RasterizationSettings(
#     image_size=image_size,
#     blur_radius=0.0,
#     faces_per_pixel=1,
# )
# rasterizer = MeshRasterizer(
#     cameras=cameras,
#     raster_settings=raster_settings
# )
# lights = PointLights(device=device, location=[[0.0, -1.0, 1.0]])
# shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

# renderer = MeshRendererWithFragments(rasterizer, shader)
# image, fragments = renderer(mesh)

# image = image.numpy()[...,:3] * 255.
# bg = np.array(tracking_imgs)
# alpha_a = (fragments.pix_to_face != -1).float().numpy()
# alpha_a *= 1.
# alpha_b = 1.
# alpha_o = alpha_a + alpha_b * (1 - alpha_a)
# img_o = (image * alpha_a + bg * alpha_b * (1 - alpha_a)) / alpha_o

# v = ((v @ c_r) * c_s)[...,:2] + c_o
# for point in v[0][...,:2]:
#     color = (0, 0, 255, 255*0.3)
#     cv2.circle(img_o[0], (int(point[0]), int(point[1])), 1, color, -1)

# cv2.imwrite('test.png', img_o[0])

# embed(); exit()

"""THIS WOKRS"""
# from viz.plot_MANO import update, MANO_FACES, update_blit
# from matplotlib.animation import FuncAnimation
# fig, ax = plt.subplots()
# ax.axis('off')
# img = ax.imshow(tracking_imgs[0])
# polygons = []
# for _ in MANO_FACES:
#     polygons.append(ax.fill([], [], 'b', alpha=0.5)[0])
# ani = FuncAnimation(fig, update_blit, frames=len(tracking_imgs), fargs=(tracking_imgs, hand_meshes), repeat=False, blit=True, interval=10)
# ani.save('animation_blit.mp4', codec="png",
#     dpi=100,
#     bitrate=-1,
#     savefig_kwargs={"transparent": True, "facecolor": "none"},)  # You may need to install ffmpeg


