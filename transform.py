import numpy as np
import torch
import torch.nn as nn

# dataset = ['HMDB', 'UCF'] #'EPIC'
dataset = ['Kinetics']
dir = './' # change to your directory
batch_size = 64  # Adjust depending on available memory
pool = nn.AdaptiveMaxPool3d((1, 1, 1))

for name in dataset:
    print(name)
    # for modality in ['clip', 'spectrogram']:
    for modality in ['spectrogram']:
        print(modality + ':')
        arr_path = f"{dir}{modality}_lst_{name}_near1.npy"
        arr = np.load(arr_path, mmap_mode='r')  # shape: (N, 1, C, D, H, W)

        if arr.shape[2] == 1:
            N = arr.shape[0]
            C = arr.shape[3]
            out_arr = np.empty((N, C, 1, 1, 1), dtype=np.float32)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = torch.tensor(arr[start:end], dtype=torch.float32)  # shape: (B, 1, C, D, H, W)
                batch = batch.squeeze(1).squeeze(1) # shape: (B, C, D, H, W)
                pooled = pool(batch)     # shape: (B, C, 1, 1, 1)
                out_arr[start:end] = pooled.cpu().numpy()
        else:
            N = arr.shape[0]
            C = arr.shape[2]
            out_arr = np.empty((N, C, 1, 1, 1), dtype=np.float32)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = torch.tensor(arr[start:end], dtype=torch.float32)  # shape: (B, 1, C, D, H, W)
                batch = batch.squeeze(1) # shape: (B, C, D, H, W)
                pooled = pool(batch)     # shape: (B, C, 1, 1, 1)
                out_arr[start:end] = pooled.cpu().numpy()

        print('output shape:', out_arr.shape)
        # Optional: save the processed output
        np.save(f"{dir}{modality}_lst_{name}_near_pooled.npy", out_arr)

    print()
