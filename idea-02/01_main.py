# Dependencies:
# pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# pip3 install gpim

import cv2
import gpim
import numpy as np

def format_data(frames):
    # The data coming in seems to be in the range 0.0 to 1.0
    # We convert it to 0..255 unsigned 8-bit integers so will
    # be saved as a black/white image.
    min_ = frames.min()
    max_ = frames.max()
    return ((frames - min_) * (1.0/(max_ - min_) * 255.0)).astype('uint8')

def sample_random(frame, percent_to_sample):
    p = percent_to_sample
    points_to_unsample = np.random.choice(a=[False, True], size=frame.shape, p=[p, 1-p])
    frame[points_to_unsample] = np.nan
    return frame

frames = np.load('../dataset/Graphene_CrSi.npy')
defects = np.load('../dataset/topo_defects.npy', allow_pickle=True)
defects = defects[()]

index = 0
# frame = format_data(frames[index])
frame = frames[index]
# cv2.imwrite('frame_%03d_01_full.png' % index, frame)
frame = sample_random(frame, 0.5)
# cv2.imwrite('frame_%03d_02_random_50.png' % index, frame)
# frame is now "sparsh"ish

# Take a smaller subregion of that frame?
sub_region = frame[0:32,0:32]

# Get full (ideal) grid indices
frame_full_grid = gpim.utils.get_full_grid(sub_region, dense_x=1)
frame_sparse_grid = gpim.utils.get_sparse_grid(sub_region)

# WUT???
# Kernel lengthscale constraints (optional)
lmin, lmax = 1., 4.
lscale = [[lmin, lmin], [lmax, lmax]] 

# Run GP reconstruction to obtain mean prediction and uncertainty for each predictied point
mean, sd, hyperparams = gpim.reconstructor(
    frame_sparse_grid,
    sub_region,
    frame_full_grid,
    # lengthscale=lscale,
    sparse=True,
    learning_rate=0.1,
    iterations=250, 
    use_gpu=True,
    verbose=False
).run()

# Plot reconstruction results
gpim.utils.plot_reconstructed_data2d(sub_region, mean, cmap='jet')
# Plot evolution of kernel hyperparameters during training
gpim.utils.plot_kernel_hyperparams(hyperparams)

print(mean)
print(sub_region)

cv2.imwrite('frame_%03d_02_random_50_reconstructed.png' % index, format_data(mean))
