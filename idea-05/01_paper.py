import time
from pathlib import Path

import cv2
import gpim
import numpy as np

class FileManager():
    def __init__(self, root: Path):
        self.root = root

    def create(self, path: str):
        file_path = self.root / Path(path)
        file_dir = file_path.parent
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        return file_path

    def get(self, path: str):
        return self.root / Path(path)


class Enhancer:
    def __init__(self):
        pass
    def enhance(self, frame):
        raise NotImplementedError

class LocalContrastEnhancer(Enhancer):
    def __init__(self):
        Enhancer.__init__(self)
        self.grid_size = (16, 16)

    def enhance(self, frame):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=self.grid_size)
        result = np.copy(frame.astype('uint8'))
        result = clahe.apply(result)
        return result


class ImageMarker:
    def __init__(self):
        pass

    def mark(self, frame, defects):
        raise NotImplementedError

class CrossDefectMarker(ImageMarker):
    def __init__(self):
        ImageMarker.__init__(self)
        self.color = (0, 0, 255)
        self.thickness = 1
        self.length = 2

    def mark(self, frame, defects):
        # def draw_defects_as_cross(frame, defects):
        rgb = np.zeros([frame.shape[0],frame.shape[1],3])
        rgb[:,:,1] = frame.astype('uint8')
        rgb[:,:,2] = frame.astype('uint8')
        rgb[:,:,0] = frame.astype('uint8')
        for polygon in defects:
            for x, y in polygon:
                cv2.line(rgb, (int(y-self.length),int(x)), (int(y+self.length),int(x)), self.color, self.thickness)
                cv2.line(rgb, (int(y),int(x-self.length)), (int(y),int(x+self.length)), self.color, self.thickness)
        return rgb

class ImageSampler:
    def __init__(self, max_percent: float):
        self.max_percent = max_percent
        self.unknown_value = np.nan  # should NOT be changed
        if self.max_percent <= 0.0:
            raise ValueError('max_percent should be > 0; it is currently: %f' % self.max_percent)

    def sample(self, frame):
        raise NotImplementedError

    @staticmethod
    def fill_unknowns(frame, value):
        mask = np.isnan(frame)
        frame[mask] = value
        return frame


class RandomSampler:
    def __init__(self, max_percent: float):
        ImageSampler.__init__(self, max_percent)

    def sample(self, frame):
        result = np.copy(frame)
        p = self.max_percent
        points_to_unsample = np.random.choice(a=[False, True], size=result.shape, p=[p, 1-p])
        result[points_to_unsample] = self.unknown_value
        return result


class SnakeSampler:
    def __init__(self, max_percent: float):
        ImageSampler.__init__(self, max_percent)

    def sample(self, frame):
        # .empty_like does *not* let you put NaNs in the array. WTF!?
        # sparse_image = np.empty_like(frame)
        sparse_image = np.empty(frame.shape)
        sparse_image.fill(self.unknown_value)
        w, h = frame.shape
        # Horizontal scan lines
        # percent_scanned = pixels_scanned / (w*h)
        # pixels_scanned = percent_scanned * (w*h)
        # pixels_scanned = w*scan_lines+h
        # percent_scanned * w * h = w*scan_lines+h
        # (percent_scanned * w * h - h) / w = scan_lines

        n_scan_lines = int((self.max_percent * w * h - h) / w)
        d = int(h / n_scan_lines)
        # print(d)
        x,y = 0,0
        while y < h-1:
            while x < w-1:
                sparse_image[y,x] = frame[y,x]
                x += 1
            for i in range(d):
                y += 1
                if y >= h:
                    y = h-1
                    break
                sparse_image[y,x] = frame[y,x]
            while x > 0:
                x -= 1
                sparse_image[y,x] = frame[y,x]

        return sparse_image


class ImageRebuilder:
    def __init__(self):
        pass

    def rebuild(self, frame):
        raise NotImplementedError


class GaussianRebuilder:
    def __init__(self):
        ImageRebuilder.__init__(self)
        
        # Kernel lengthscale constraints should be close to the scale
        # of the features you are interested in.
        # Lower numbers = "sharper image"
        # Higher numbers = "blurrier image"
        # For the data we are working with - the atomic structures are ~10-16px
        # so it makes sense to use those as the lmin/max parameters; howver, 
        # the division of the images into smaller chunks makes some parts
        # seem better/worse (clear/not) than others.
        self.lmin, self.lmax = 10.0, 16.0

        # We divide the full image into 64x64 sub images.
        # Larger subimages take longer to compute and do not
        # show an improvement in quality.
        self.subimage_size = (64, 64)
        
        # We pad the subimages to improve the quality.
        # Without appropriate padding there is a clear line between
        # one frame and another. I recommend setting the padding to
        # the lmax of the Gaussian reconstruction process.
        self.padding_size = (16, 16)
        
    
    def rebuild(self, frame):
        reconstructed_image = np.zeros_like(frame)
        padded_frame = np.pad(frame, self.padding_size)

        w, h = frame.shape
        subw, subh = self.subimage_size
        xcount = int(w//subw)
        ycount = int(h//subh)
        for y in range(ycount):
            for x in range(xcount):
                # print(f'{time.time()//60} -- y: {y} of {ycount}; x: {x} of {xcount};')
                # 0,0 is the origin of the padded image
                # +padding is the origin of the region of interest
                # +padding+width is the end of the region of interest
                # +padding+width+padding is the end of the padded image

                rx_start = x*subw
                rx_end = x*subw+subw

                ry_start = y*subw
                ry_end = y*subw+subw

                sub_image = padded_frame[
                    rx_start:rx_end+2*self.padding_size[0],
                    ry_start:ry_end+2*self.padding_size[0]
                ]
                # sub_image = sparse_image[x*subw:x*subw+subw,y*subh:y*subh+subh]
                indexes_full = gpim.utils.get_full_grid(sub_image, dense_x=1)
                indexes_sparse = gpim.utils.get_sparse_grid(sub_image)

                lscale = [[self.lmin, self.lmin], [self.lmax, self.lmax]] 

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
                    self.padding_size[0]:self.padding_size[0]+subw,
                    self.padding_size[1]:self.padding_size[1]+subh
                ]

                # Plot reconstruction results and then
                # Plot evolution of kernel hyperparameters during training
                # gpim.utils.plot_reconstructed_data2d(sub_image, mean, cmap='jet')
                # gpim.utils.plot_kernel_hyperparams(hyperparams)

        return reconstructed_image


class FourierRebuilder(ImageRebuilder):
    def __init__(self):
        ImageRebuilder.__init__(self)
        self.fmin = 0
        self.fmax = 56

    def rebuild(self, frame):
        # Note: we probably should keep the image with NaNs in a safe
        # place and then restore the corrected image to only the
        # pixels with NaNs.
        img = np.copy(frame)
        img = ImageSampler.fill_unknowns(img, 0)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        # print(dft)
        # cv2.imwrite('fft_0_dft.png', format_data(dft))
        # cv2.imwrite('fft_1_dft_shift.png', format_data(dft_shift))
        # cv2.imwrite('fft_2_magnitude.png', format_data(magnitude_spectrum))

        rows, cols = img.shape
        rmid, cmid = int(rows/2), int(cols/2)
        
        # create a "ring" of 1's for which frequencies we want to keep
        mask = np.zeros((rows,cols,1), np.uint8)
        cv2.circle(mask, (rmid, cmid), self.fmax, 1, -1)
        cv2.circle(mask, (rmid, cmid), self.fmin, 0, -1)
        # mask[rmid-fmax:rmid+fmax, cmid-fmax:cmid+fmax] = 1
        # mask[rmid-fmin:rmid+fmin, cmid-fmin:cmid+fmin] = 0
        # cv2.imwrite('fft_3_mask.png', format_data(mask.astype('uint8')))

        # apply mask and inverse DFT
        fshift = dft_shift*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        # cv2.imwrite('fft_4_fshift.png', format_data(fshift))
        # cv2.imwrite('fft_5_f_ishift.png', format_data(f_ishift))
        # cv2.imwrite('fft_6_img_back.png', format_data(img_back))
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        # cv2.imwrite('fft_7_out.png', format_data(img_back))

        return format_data(img_back)


# This is kinda sloppy :(
def format_data(frames):
    # The data coming in seems to be in the range 0.0 to 1.0
    # We convert it to 0..255 unsigned 8-bit integers so will
    # be saved as a black/white image.
    min_ = frames.min()
    max_ = frames.max()
    return ((frames - min_) * (1.0/(max_ - min_) * 255.0)).astype('uint8')

# Load the data
files = FileManager(Path('.'))
frames = np.load(files.get('dataset/Graphene_CrSi.npy'))
frames = format_data(frames)
defects = np.load(files.get('dataset/topo_defects.npy'), allow_pickle=True)
defects = defects[()]

# Choose your frame
index = 20

# Sample at various percents
# Due to the way we implemented snaking it has descrete locations
# that are all the same. So we limit the value we look at to the
# unique ones. Also note:
# 0% makes no sense to rebuild.
# 1% seems to fail for gaussian reconstruction.
targets = list(range(2,14)) + [16,18,25,33,50]

for i in targets:
    sampler = SnakeSampler(float(i)/100.0)
    s_name = 'snake'
    # rebuilder = GaussianRebuilder()
    # r_name = 'gaussian'
    rebuilder = FourierRebuilder()
    r_name = 'fourier'
    marker = CrossDefectMarker()
    enhancer = LocalContrastEnhancer()

    img = sampler.sample(frames[index])
    start = time.time()
    img = rebuilder.rebuild(img)
    end = time.time()
    print('%03d - %0.2f' % (i, end-start))
    
    img = enhancer.enhance(img)

    cv2.imwrite(files.create('results/%03d_%s_%03d_%s_clean.png' % (
        index,
        s_name,
        i,
        r_name
    ) ).as_posix(), img)

    img = marker.mark(img, defects[index])

    cv2.imwrite(files.create('results/%03d_%s_%03d_%s_marked.png' % (
        index,
        s_name,
        i,
        r_name
    ) ).as_posix(), img)
