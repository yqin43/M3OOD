from dataloader_video_flow import *
import numpy as np
from mmaction.apis import init_recognizer
import torch
import torch.nn as nn
import os
from tqdm import tqdm

near_ood = True
d_path = {
    'HMDB': '/data/shared_dataset/data/HMDB51/',
    'UCF': '/data/shared_dataset/data/UCF101/',
    'EPIC': '/data/shared_dataset/data/EPIC_KITCHENS/',
    'Kinetics': '/data/shared_dataset/data/Kinetics-600/'
}

config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'

device = 'cuda:3' # or 'cpu'
device = torch.device(device)

v_dim = 2304
f_dim = 2048

# dataset_lst = ['HMDB','UCF','Kinetics']
# dataset_lst = ['EPIC']
# dataset_lst = ['HMDB','Kinetics'] # far test
dataset_lst = ['UCF'] # far test 2

for d in dataset_lst:
    dataset = d
    if near_ood:
        if dataset == 'HMDB':
            num_class = 25
        elif dataset == 'UCF':
            num_class = 50
        elif dataset == 'Kinetics':
            num_class = 129
        elif dataset == 'EPIC':
            num_class = 4
    else:
        if dataset == 'HMDB':
            num_class = 43
        elif dataset == 'Kinetics':
            num_class = 229
    # num_class = 25

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

    train_dataset = EPICDOMAIN(split='test', eval=True, cfg=cfg, cfg_flow=cfg_flow, datapath=d_path[dataset], dataset=dataset, near_ood=near_ood, far_ood=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=16, shuffle=False,
                                                    pin_memory=(device.type == "cuda"), drop_last=False)

    # If you kept the original mmaction2 pipelines
    clip_sample, flow_sample, label = train_dataset[0]
    # print(list(clip_sample.keys()))
    # print(list(flow_sample.keys()))
    print("RGB tensor shape :", clip_sample['video'].shape)       # e.g. (C, T, H, W)
    print("Flow tensor shape:", flow_sample['video'].shape)        # e.g. (C, T, H, W)
    exit(0)
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

    np.save(os.path.join(output_dir, f"clip_mf_test_{dataset}_{ood_str}_pooled.npy"), clip_pooled_arr)
    np.save(os.path.join(output_dir, f"spectrogram_mf_test_{dataset}_{ood_str}_pooled.npy"), spectrogram_pooled_arr)

    print("Saved shapes:", clip_pooled_arr.shape, spectrogram_pooled_arr.shape)


# clip_lst1 = []
# spectrogram_lst1 = []
# for clip, spectrogram, labels in train_dataloader:
#     clip_lst1.append(clip['imgs'].numpy())
#     spectrogram_lst1.append(spectrogram['imgs'].numpy())

# clip_lst = np.stack(clip_lst1)
# spectrogram_lst = np.stack(spectrogram_lst1)

# # np.save(f'clip_lst_{dataset}_{ood_str}.npy', clip_lst)
# # np.save(f'spectrogram_lst_{dataset}_{ood_str}.npy', spectrogram_lst)

# batch_size = 64
# dir = './'
# pool = nn.AdaptiveMaxPool3d((1, 1, 1))
# arr_lst = [clip_lst, spectrogram_lst]
# for modality in ['spectrogram','clip']:
#     arr = arr_lst[0] if modality == 'clip' else arr_lst[1]
#     if arr.shape[2] == 1:
#             N = arr.shape[0]
#             C = arr.shape[3]
#             out_arr = np.empty((N, C, 1, 1, 1), dtype=np.float32)

#             for start in range(0, N, batch_size):
#                 end = min(start + batch_size, N)
#                 batch = torch.tensor(arr[start:end], dtype=torch.float32)  # shape: (B, 1, C, D, H, W)
#                 batch = batch.squeeze(1).squeeze(1) # shape: (B, C, D, H, W)
#                 pooled = pool(batch)     # shape: (B, C, 1, 1, 1)
#                 out_arr[start:end] = pooled.cpu().numpy()
#     else:
#         N = arr.shape[0]
#         C = arr.shape[2]
#         out_arr = np.empty((N, C, 1, 1, 1), dtype=np.float32)

#         for start in range(0, N, batch_size):
#             end = min(start + batch_size, N)
#             batch = torch.tensor(arr[start:end], dtype=torch.float32)  # shape: (B, 1, C, D, H, W)
#             batch = batch.squeeze(1) # shape: (B, C, D, H, W)
#             pooled = pool(batch)     # shape: (B, C, 1, 1, 1)
#             out_arr[start:end] = pooled.cpu().numpy()

#     print('output shape:', out_arr.shape)
#     # Optional: save the processed output
#     np.save(f"{dir}{modality}_lst_{dataset}_far_pooled.npy", out_arr)
# print('file saved.')