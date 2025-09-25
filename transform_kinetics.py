import os
import numpy as np
import torch
import torch.nn as nn

input_dir = './' # change to your directory
output_path = './' # change to your directory

batch_prefix = 'clip_batch_'
num_batches = 704  # 0000 to 0704 inclusive
batch_size = 64

# First pass: determine total number of samples
total_samples = 0
for i in range(num_batches + 1):
    path = os.path.join(input_dir, f'{batch_prefix}{i:04d}.npy')
    arr = np.load(path, mmap_mode='r')
    total_samples += arr.shape[0]
    del arr

# Create memory-mapped output array
C = 3
pooled_shape = (total_samples, C, 1, 1, 1)
pooled_out = np.lib.format.open_memmap(output_path, dtype=np.float32, mode='w+', shape=pooled_shape)

# Define pooling layer
pool = nn.AdaptiveMaxPool3d((1, 1, 1))

# Second pass: process and fill pooled output
write_idx = 0
for i in range(num_batches + 1):
    path = os.path.join(input_dir, f'{batch_prefix}{i:04d}.npy')
    print(f'Processing {path}')
    arr = np.load(path, mmap_mode='r')  # shape: (B, 1, 3, 32, 224, 224)
    B = arr.shape[0]
    print(arr.shape)
    # exit()
    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        # print(start, end)
        batch = torch.tensor(arr[start:end], dtype=torch.float32)
        batch = batch.squeeze(1) # (B, 3, 32, 224, 224)
        pooled = pool(batch)  # (B, 3, 1, 1, 1)
        pooled_np = pooled.cpu().numpy()
        pooled_out[write_idx:write_idx + (end - start)] = pooled_np
        write_idx += (end - start)

    del arr  # free mmap memory
    print(f'Done batch {i:04d}, total written: {write_idx}')

print(f'Final output saved at: {output_path}, shape: {pooled_out.shape}')
