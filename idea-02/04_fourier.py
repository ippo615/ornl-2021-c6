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


def filter_frequencies(frame, fmin, fmax):
    img = frame
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    # print(dft)
    # cv2.imwrite('fft_0_dft.png', format_data(dft))
    # cv2.imwrite('fft_1_dft_shift.png', format_data(dft_shift))
    cv2.imwrite('fft_2_magnitude.png', format_data(magnitude_spectrum))

    rows, cols = img.shape
    rmid, cmid = int(rows/2), int(cols/2)
    
    # create rectangluar "ring" of 1's for which frequencies we want to keep
    # probably should be a circle
    mask = np.zeros((rows,cols,1), np.uint8)
    cv2.circle(mask, (rmid, cmid), 14*4, 1, -1)
    # mask[rmid-fmax:rmid+fmax, cmid-fmax:cmid+fmax] = 1
    # mask[rmid-fmin:rmid+fmin, cmid-fmin:cmid+fmin] = 0
    print(mask.shape)
    cv2.imwrite('fft_3_mask.png', format_data(mask.astype('uint8')))

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    # cv2.imwrite('fft_4_fshift.png', format_data(fshift))
    # cv2.imwrite('fft_5_f_ishift.png', format_data(f_ishift))
    # cv2.imwrite('fft_6_img_back.png', format_data(img_back))
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    cv2.imwrite('fft_7_out.png', format_data(img_back))

    return img_back


frames = np.load('../dataset/Graphene_CrSi.npy')
defects = np.load('../dataset/topo_defects.npy', allow_pickle=True)
defects = defects[()]

def process_image(frame, sample_percent, fmin, fmax):
    img = np.copy(frame)
    sparse_image = fill_nan_with(sample_random(img, sample_percent), 0.0)
    return filter_frequencies(sparse_image, fmin, fmax)


index = 0
# frame = format_data(frames[index])
#frame = frames[index][256:512, 256:512]
frame = frames[index]
import time
start = time.time()
fmin = 0
fmax = 100
reconstruction = process_image(frame, 0.2, fmin, fmax)
cv2.imwrite('z_frame_%03d_%04d_%04d.png' % (index, fmin, fmax), format_data(reconstruction))
# for fmin in range(0,10):
#     for fmax in range(fmin,64,1):
#         reconstruction = process_image(frame, 0.5, fmin, fmax)
#         cv2.imwrite('z_frame_%03d_%04d_%04d.png' % (index, fmin, fmax), format_data(reconstruction))
#         print('frame_%03d_%04d_%04d.png' % (index, fmin, fmax))
end = time.time()
print('Duration %f seconds' % (end-start))
# cv2.imwrite('frame_%03d_02_random_50_reconstructed.png' % index, format_data(reconstruction))

