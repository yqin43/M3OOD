import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from mmaction.apis import init_recognizer
# from dataloader_video_flow import EPICDOMAIN
from dataloader_video_flow_hac import HACDOMAIN
import numpy as np
import cv2
from skimage.feature import graycomatrix
from math import pi
from tqdm import tqdm

EPS = 1e-8

# ---------------- GPU‐compatible statistics ----------------

def skew_t(x, dim=0):
    mu  = x.mean(dim=dim, keepdim=True)
    sig = x.std(dim=dim, unbiased=False, keepdim=True)
    return ((x - mu)**3).mean(dim=dim) / (sig**3 + EPS)

def kurtosis_t(x, dim=0):
    mu  = x.mean(dim=dim, keepdim=True)
    sig = x.std(dim=dim, unbiased=False, keepdim=True)
    return ((x - mu)**4).mean(dim=dim) / (sig**4 + EPS) - 3

def iqr_t(x):
    # detach → CPU NumPy → percentile → back to tensor
    arr = x.detach().cpu().numpy()
    return torch.tensor(
        np.percentile(arr, 75) - np.percentile(arr, 25),
        device=x.device,
        dtype=x.dtype
    )

def gini_t(x):
    if x.min() < 0:
        x = x - x.min()
    xs, _ = x.sort()
    n = x.numel()
    idx = torch.arange(1, n+1, device=x.device, dtype=x.dtype)
    return (2*(idx*xs).sum()/(xs.sum()+EPS) - n - 1)/n

def extra_clip_stats_t(gray, mag):
    # intensity branch
    muI, stdI = gray.mean(), gray.std(unbiased=False)
    skewI, kurtI = skew_t(gray), kurtosis_t(gray)
    mnI, mxI = gray.min(), gray.max()
    medI = gray.median()
    iqrI = iqr_t(gray)
    giniI = gini_t(gray)
    madI  = (gray - medI).abs().median()
    aadI  = (gray - medI).abs().mean()
    cvI   = stdI/(muI + EPS)

    # flow branch: compute quantiles via NumPy to avoid large-tensor quantile
    mag_np = mag.detach().cpu().numpy()
    q01_np, q99_np = np.quantile(mag_np, [0.01, 0.99])
    q01 = torch.tensor(q01_np, device=mag.device)
    q99 = torch.tensor(q99_np, device=mag.device)

    skewM, kurtM = skew_t(mag), kurtosis_t(mag)
    iqrM        = iqr_t(mag)
    cvM         = mag.std(unbiased=False)/(mag.mean()+EPS)
    p1M         = ((mag < q01) | (mag > q99)).float().mean()
    m, s        = mag.mean(), mag.std(unbiased=False)
    p3sM        = ((mag < m-3*s) | (mag > m+3*s)).float().mean()

    values = [
        muI,   stdI,   skewI,   kurtI,
        mnI,   mxI,    medI,    iqrI,
        giniI, madI,   aadI,    cvI,
        skewM, kurtM,  iqrM,    cvM,
        p1M,   p3sM
    ]
    # make sure every tensor is 0-d before stacking
    values = [v.squeeze() for v in values]
    stats = torch.stack(values)
    return stats[:16]

def clip_stats_t(frames):
    # frames: (T,3,H,W)
    mean_rgb = frames.mean(dim=(0,2,3))                            # (3,)
    var_rgb  = frames.var(dim=(0,2,3), unbiased=False)             # (3,)
    R, G, B  = frames[:,0], frames[:,1], frames[:,2]              # each (T,H,W)
    rg = (R - G).abs()
    yb = (0.5*(R + G) - B).abs()
    std_rg, std_yb = rg.std(unbiased=False), yb.std(unbiased=False)
    mean_rg, mean_yb = rg.mean(), yb.mean()
    col = (std_rg**2 + std_yb**2).sqrt() + 0.3*(mean_rg**2 + mean_yb**2).sqrt()
    return torch.cat([mean_rgb, var_rgb, col.unsqueeze(0)], dim=0)

def edge_density_t(frames):
    # frames: (T,3,H,W)
    edges = []
    for t in range(frames.shape[0]):
        img = frames[t].permute(1,2,0).cpu().numpy()  # (H,W,3)
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        ed = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
        edges.append(ed.mean())
    return torch.tensor([float(np.mean(edges))], device=frames.device)

def texture_stats_t(frames):
    ents = []
    for t in range(frames.shape[0]):
        img = frames[t].permute(1,2,0).cpu().numpy()
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        G = graycomatrix(gray, distances=[1], angles=[0], levels=256,
                         symmetric=True, normed=True)[:,:,0,0]
        p = G[G>0]/G.sum()
        ents.append(-(p*np.log2(p)).sum())
    return torch.tensor([float(np.mean(ents))], device=frames.device)

def flow_stats_t(flow):
    u, v = flow[:,0], flow[:,1]         # (T-1,H,W)
    mag   = torch.hypot(u, v).flatten()
    mean_mag = mag.mean()
    var_mag  = mag.var(unbiased=False)
    theta    = torch.atan2(v, u).flatten()
    bins     = torch.linspace(-pi, pi, 9, device=flow.device)
    inds     = torch.bucketize(theta, bins)-1
    inds     = inds.clamp(0,7)
    hof      = torch.zeros(8, device=flow.device).scatter_add(0, inds, mag)
    hof      = hof/(hof.sum()+EPS)
    return torch.cat([mean_mag.unsqueeze(0), var_mag.unsqueeze(0), hof], dim=0)

# ---------------- data conversion ----------------

def get_frames_gpu(sample, device):
    arr = sample['imgs'].to(device)            # [1,T-1,C,H,W] or [T-1,C,H,W]
    arr = arr.squeeze(0).squeeze(0) if arr.dim() == 6 else arr.squeeze(0)
    if arr.ndim == 4 and arr.shape[0] == 3 and arr.shape[1] != 3:
        # frames is (3, T, H, W): swap axis 0 and 1
        arr = arr.permute(1, 0, 2, 3)
    return arr.float()/255.0

def get_flow_gpu(sample, device):
    arr = sample['imgs'].to(device)            # [1,T-1,C,H,W] or [T-1,C,H,W]
    arr = arr.squeeze(0).squeeze(0) if arr.dim() == 6 else arr.squeeze(0)
    if arr.ndim == 4 and arr.shape[0] == 3 and arr.shape[1] != 3:
        # frames is (3, T, H, W): swap axis 0 and 1
        arr = arr.permute(1, 0, 2, 3)
    return arr.float()

# ---------------- feature extraction ----------------

def build_meta_vectors(dataloader, device):
    feats, labels = [], []
    for clip_s, flow_s, label in tqdm(dataloader, desc='extracting meta-features'):
        frames = get_frames_gpu(clip_s, device)  # (T,3,H,W)
        flow   = get_flow_gpu(flow_s, device)    # (T-1,2,H,W)

        # basic + base stats (26 dims)
        T, C, H, W    = frames.shape
        Tf, Cf, Hf, Wf = flow.shape
        basic = torch.tensor([T, H, W, H/W, Hf, Wf, Hf/Wf],
                             device=device)
        rgb_s  = clip_stats_t(frames)               # (7,)
        edge_s = edge_density_t(frames)             # (1,)
        tex_s  = texture_stats_t(frames)            # (1,)
        flow_s = flow_stats_t(flow)                 # (10,)

        gray = (0.2989*frames[:,0] + 0.5870*frames[:,1] + 0.1140*frames[:,2]).flatten()
        mag  = torch.hypot(flow[:,0], flow[:,1]).flatten()
        ext16= extra_clip_stats_t(gray, mag)        # (16,)

        feat = torch.cat([basic, rgb_s, edge_s, tex_s, flow_s, ext16], dim=0)  # (42,)
        feats.append(feat)
        labels.append(label.item())

    X = torch.stack(feats).cpu().numpy()  # (N,42)
    y = np.array(labels)                  # (N,)
    return X, y

# ----------------------------- main ------------------------------------

if __name__ == '__main__':
    near_ood = False
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    v_dim, f_dim = 2304, 2048

    d_path = {
        'HMDB': '/data/shared_dataset/data/HMDB51/',
        'UCF': '/data/shared_dataset/data/UCF101/',
        'EPIC': '/data/shared_dataset/data/EPIC_KITCHENS/',
        'Kinetics': '/data/shared_dataset/data/Kinetics-600/'
    }
    config_file      = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'

    dataset_lst = ['HMDB']#, 'Kinetics']
    for dataset in dataset_lst:
        num_class = {'HMDB':25,'UCF':50,'Kinetics':129,'EPIC':4}[dataset] \
                    if near_ood else {'HMDB':43,'Kinetics':229}[dataset]
        print(f"Dataset {dataset}, num_class = {num_class}")

        model = init_recognizer(config_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(v_dim, num_class).to(device)
        model = nn.DataParallel(model)

        mflow = init_recognizer(config_file_flow, device=device, use_frames=True)
        mflow.cls_head.fc_cls = nn.Linear(f_dim, num_class).to(device)
        mflow = nn.DataParallel(mflow)
        # HAC
        ds = HACDOMAIN(cfg=model.module.cfg, cfg_flow=mflow.module.cfg, datapath='/data/shared_dataset/data/HAC/')
        # ds = EPICDOMAIN(
        #     split='test', eval=True,
        #     cfg=model.module.cfg,
        #     cfg_flow=mflow.module.cfg,
        #     datapath=d_path['UCF'], # ucf debug
        #     dataset= dataset,
        #     ood_dataset='UCF', # ucf debug
        #     near_ood=near_ood,
        #     far_ood=True
        # )
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=16, pin_memory=True)

        X, y = build_meta_vectors(loader, device)
        np.save(f'meta_features_HAC.npy', X)
        np.save(f'meta_labels_HAC.npy', y)
        print('saved:', X.shape, y.shape)
