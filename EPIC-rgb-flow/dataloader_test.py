from dataloader_video_flow_epic import *
import numpy as np
from tqdm import tqdm
from mmaction.apis import init_recognizer
import torch
import torch.nn as nn
import os

near_ood = False
dataset_lst = ['HMDB', 'EPIC', 'Kinetics']
dataset = dataset_lst[2]

config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'

device = 'cuda:5' # or 'cpu'
device = torch.device(device)

v_dim = 2304
f_dim = 2048

if not near_ood:
    if dataset == 'HMDB':
        num_class = 43
    elif dataset == 'Kinetics':
        num_class = 229
else:
    num_class = 4
# num_class = 4

print("####",num_class)
# build the model from a config file and a checkpoint file
model = init_recognizer(config_file, device=device, use_frames=True)
model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()
cfg = model.cfg
model = torch.nn.DataParallel(model)

model_flow = init_recognizer(config_file_flow, device=device,use_frames=True)
model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
cfg_flow = model_flow.cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = EPICDOMAIN(split='test', eval=True, cfg=cfg, cfg_flow=cfg_flow, datapath='/data/shared_dataset/data/EPIC_KITCHENS/', far_ood=True)
batch_size = 32  # or whatever fits in your GPU memory
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=12, shuffle=False,
    pin_memory=(device.type == "cuda"), drop_last=False
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=16, shuffle=False,
                                                    pin_memory=(device.type == "cuda"), drop_last=False)

# try to debug
batch_size = 64
output_dir = './'
pool = nn.AdaptiveMaxPool3d((1, 1, 1))

clip_pooled_list = []
spectrogram_pooled_list = []

for clip, spectrogram, _ in tqdm(train_dataloader, desc='Pooling Batches'):
    for data, pooled_list in zip(
        [clip['imgs'], spectrogram['imgs']],
        [clip_pooled_list, spectrogram_pooled_list]
    ):
        x = data.squeeze(1) if data.shape[2] != 1 else data.squeeze(1).squeeze(1)  # -> (B, C, D, H, W)
        pooled = pool(x.float())  # -> (B, C, 1, 1, 1)
        pooled_list.append(pooled.cpu().numpy())

# Stack and save after the loop
clip_pooled_arr = np.concatenate(clip_pooled_list, axis=0)
spectrogram_pooled_arr = np.concatenate(spectrogram_pooled_list, axis=0)

if near_ood:
    ood_str = 'near'
else:
    ood_str = 'far'
np.save(os.path.join(output_dir, f"clip_test_{dataset}_EPIC_{ood_str}_pooled.npy"), clip_pooled_arr)
np.save(os.path.join(output_dir, f"spectrogram_test_{dataset}_EPIC_{ood_str}_pooled.npy"), spectrogram_pooled_arr)

print("Saved shapes:", clip_pooled_arr.shape, spectrogram_pooled_arr.shape)

