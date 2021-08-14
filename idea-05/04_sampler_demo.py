import json
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
# frames = format_data(frames)
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

targets = [12, 25, 33, 50]
x_start, x_end = 800, 928
y_start, y_end = 800, 928

demo_image = None
for i in targets:

    # Full image
    img = np.copy(frames[index])
    row = img[y_start:y_end, x_start:x_end]

    # Random Sample
    sampler = RandomSampler(float(i)/100.0)
    img = np.copy(frames[index])
    img = sampler.sample(img)
    img = ImageSampler.fill_unknowns(img, 0.0)
    row = np.hstack((row, img[y_start:y_end, x_start:x_end]))
    
    # Snaking
    sampler = SnakeSampler(float(i)/100.0)
    img = np.copy(frames[index])
    img = sampler.sample(img)
    img = ImageSampler.fill_unknowns(img, 0.0)
    row = np.hstack((row, img[y_start:y_end, x_start:x_end]))

    if demo_image is not None:
        demo_image = np.vstack((demo_image, row))
    else:
        demo_image = row

enhancer = LocalContrastEnhancer()
cv2.imwrite('samples.png', enhancer.enhance(format_data(demo_image)))
