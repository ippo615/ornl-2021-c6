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

def fill_nan_with(arr, value):
    mask = np.isnan(arr)
    arr[mask] = value
    return arr

def process_image(frame, subimage_size, padding_size, sample_percent):
    reconstructed_image = np.zeros_like(frame)
    sparse_image = sample_random(frame, sample_percent)

    padded_frame = np.pad(sparse_image, padding_size)

    w, h = frame.shape
    subw, subh = subimage_size
    xcount = int(w//subw)
    ycount = int(h//subh)
    for y in range(ycount):
        for x in range(xcount):
            print(f'{time.time()//60} -- y: {y} of {ycount}; x: {x} of {xcount};')
            # 0,0 is the origin of the padded image
            # +padding is the origin of the region of interest
            # +padding+width is the end of the region of interest
            # +padding+width+padding is the end of the padded image

            rx_start = x*subw
            rx_end = x*subw+subw

            ry_start = y*subw
            ry_end = y*subw+subw

            sub_image = padded_frame[rx_start:rx_end+2*padding_size[0], ry_start:ry_end+2*padding_size[0]]
            # sub_image = sparse_image[x*subw:x*subw+subw,y*subh:y*subh+subh]
            indexes_full = gpim.utils.get_full_grid(sub_image, dense_x=1)
            indexes_sparse = gpim.utils.get_sparse_grid(sub_image)

            # Kernel lengthscale constraints (optional)
            # Lower numbers = "sharper image"
            # Higher numbers = "blurrier image"
            # For the data we are working with - the atomic structures are ~10-16px
            # so it makes sense to use those as the lmin/max parameters; howver, 
            # the division of the images into smaller chunks makes some parts
            # seem better/worse (clear/not) than others.
            lmin, lmax = 10.0, 16.0
            # lmin, lmax = 1.0, 2.0
            lscale = [[lmin, lmin], [lmax, lmax]] 

            mean, sd, hyperparams = gpim.reconstructor(
                indexes_sparse,
                sub_image,
                indexes_full,
                lengthscale=lscale,
                sparse=True,
                learning_rate=0.1,
                iterations=2, 
                use_gpu=True,
                verbose=False
            ).run()

            # For debugging/sanity
            # reconstructed_image[rx_start:rx_end,ry_start:ry_end] = fill_nan_with(sub_image[
            #     padding_size[0]:padding_size[0]+subw,
            #     padding_size[1]:padding_size[1]+subh
            # ], 0.0)
            reconstructed_image[rx_start:rx_end,ry_start:ry_end] = mean[
                padding_size[0]:padding_size[0]+subw,
                padding_size[1]:padding_size[1]+subh
            ]

            # Plot reconstruction results and then
            # Plot evolution of kernel hyperparameters during training
            # gpim.utils.plot_reconstructed_data2d(sub_image, mean, cmap='jet')
            # gpim.utils.plot_kernel_hyperparams(hyperparams)

    return reconstructed_image

frames = np.load('../dataset/Graphene_CrSi.npy')
defects = np.load('../dataset/topo_defects.npy', allow_pickle=True)
defects = defects[()]

index = 0
# frame = format_data(frames[index])
frame = frames[index]
import time
start = time.time()
# reconstruction = process_image(frame[0:128,0:512], (64,64), (16, 16), 0.5)
reconstruction = process_image(frame, (64,64), (16, 16), 0.6)
end = time.time()
print('Duration %f seconds' % (end-start))
# (a) takes 774 seconds
# reconstruction = process_image(frame[0:128,0:512], (32,32), (8, 8), 0.5)
# (b) takes 4903 seconds (82 min)
# reconstruction = process_image(frame[0:128,0:512], (64,64), (16, 16), 0.5)
cv2.imwrite('frame_%03d_02_random_60_reconstructed.png' % index, format_data(reconstruction))

# Reconstruction of the entire image in 2 iterations;
# data (%) - duration (s)
# 10 - 80s
# 20 - 279s
# 30 - 411s
# 40 - 940s 
# 50 - ??
