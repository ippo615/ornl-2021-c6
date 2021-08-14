import json
import time
from pathlib import Path
from collections import defaultdict

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
        rgb = np.copy(frame)
        # rgb = np.zeros([frame.shape[0],frame.shape[1],3])
        # rgb[:,:,1] = frame.astype('uint8')
        # rgb[:,:,2] = frame.astype('uint8')
        # rgb[:,:,0] = frame.astype('uint8')
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


class RegionDescriptor:
    def __init__(self):
        self.region_size = (64,64)
        self.output_size = (8,8)
        self.summary = {}
        self._summary_vectors = {}
        self._results = {
            'actual_defects': 0,
            'correctly_predicted_defects': 0,
            'missed_defects': 0
        }

    def compute_at(self, img, x, y):
        pass

    def summarize_result(self, result):
        hw_out = int(self.output_size[0]//2)
        hh_out = int(self.output_size[1]//2)
        s_start_x = int(self.region_size[0]/2) - hw_out
        s_start_y = int(self.region_size[1]/2) - hh_out
        s_end_x = int(self.region_size[0]/2) + hw_out
        s_end_y = int(self.region_size[1]/2) + hh_out
        return ''.join(''.join(['1' if i == 255 else '0' for i in part[s_start_x:s_end_x]]) for part in result[s_start_y:s_end_y])

    def load_summary(self, summary):
        self.summary = summary
        for s in summary:
            self._summary_vectors[s] = self._summary_to_vector(s)

    def _summary_to_vector(self, summary):
        values = np.fromiter(summary, dtype=np.int, count=len(summary))
        length = self.output_size[0]*self.output_size[1]
        vector = np.zeros([length])
        vector[:values.shape[0]] = values
        return vector

    def classify_summary(self, summary):
        # If there is an exact match
        if summary in self.summary:
            return self.summary[summary]
        
        # Find the closest summaries
        min_distance = 1+len(summary)*len(summary)
        distances = defaultdict(list)
        target = self._summary_to_vector(summary)
        for s in self.summary:
            other = self._summary_vectors[s]
            distance = np.linalg.norm(other - target)
            distances[distance].append(s)
            if distance < min_distance:
                min_distance = distance
        closest_summaries = distances[min_distance]

        # Use the sum of defects and normals of the closest
        # summaries to "vote" for what this is.
        defects = 0
        normals = 0
        for s in closest_summaries:
            defects += self.summary[s]['defect_count']
            normals += self.summary[s]['normal_count']
        
        # Most votes wins
        return {
            'defect_count': defects,
            'normal_count': normals
        }

    def compute_all(self, img):
        w, h = img.shape
        results = []
        for y in range(0, h, self.region_size[1]):
            for x in range(0, w, self.region_size[0]):
                results.append([[x,y], self.compute_at(img, x,y)])
        return np.array(results)

    def classify_all_as_image_with_defects(self, img, defects):
        results = np.zeros_like(img)
        summaries = {}
        w, h = img.shape
        rgb = np.zeros([img.shape[0],img.shape[1],3])
        
        correct_classifications = 0
        false_positives = 0
        false_negatives = 0
        guesses = 0

        for y in range(0, h, self.output_size[1]):
            for x in range(0, w, self.output_size[0]):

                center_x = x+int(self.region_size[0]//2)
                center_y = y+int(self.region_size[1]//2)
                hw_out = int(self.output_size[0]//2)
                hh_out = int(self.output_size[1]//2)
                start_x = center_x - hw_out
                start_y = center_y - hh_out
                end_x = center_x + hw_out
                end_y = center_y + hh_out
                s_start_x = int(self.region_size[0]/2) - hw_out
                s_start_y = int(self.region_size[1]/2) - hh_out
                s_end_x = int(self.region_size[0]/2) + hw_out
                s_end_y = int(self.region_size[1]/2) + hh_out

                has_defect = False
                for polygon in defects:
                    for defect_y, defect_x in polygon:
                        if start_x <= defect_x <= end_x and start_y <= defect_y <= end_y:
                            has_defect = True
                            break
                    if has_defect:
                        break

                result = self.compute_at(img, x, y)
                summary = self.summarize_result(result)
                classification = self.classify_summary(summary)
                class_defect = classification['defect_count'] > classification['normal_count']
                class_normal = not class_defect
                class_mixed = (classification['defect_count'] != 0 and  classification['normal_count'] != 0)

                try:
                    if has_defect:
                        self._results['actual_defects'] += 1
                        if class_defect:
                            self._results['correctly_predicted_defects'] += 1
                        if not class_defect:
                            self._results['missed_defects'] += 1

                    # Remember opencv is BGR order
                    if class_defect:
                        rgb[start_y:end_y, start_x:end_x, 0] = 255
                        rgb[start_y:end_y, start_x:end_x, 1] = 255
                        rgb[start_y:end_y, start_x:end_x, 2] = 255

                    # # Correct matches are blue
                    # if class_defect and has_defect:
                    #     rgb[start_y:end_y, start_x:end_x, 0] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                    #     correct_classifications += 1
                    # elif class_normal and not has_defect:
                    #     rgb[start_y:end_y, start_x:end_x, 0] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                    #     correct_classifications += 1
                    
                    # # Mixed things are purple:
                    # if class_mixed:
                    #     guesses += 1
                    #     rgb[start_y:end_y, start_x:end_x, 0] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                    #     rgb[start_y:end_y, start_x:end_x, 2] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                    
                    # # Errors are red:
                    # if class_defect and not has_defect:
                    #     rgb[start_y:end_y, start_x:end_x, 2] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                    #     false_positives += 1
                    # elif class_normal and has_defect:
                    #     rgb[start_y:end_y, start_x:end_x, 2] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                    #     false_negatives += 1

                except ValueError:
                    # Forget the ones at the end that dont fit
                    #print('%d, %d' % (x, y))
                    pass

        with open('test_summary', 'a') as f:
            f.write('Classification Stats:\n')
            f.write('correct_classifications: %5d\n' % correct_classifications)
            f.write('false_positives: %5d\n' % false_positives)
            f.write('false_negatives: %5d\n' % false_negatives)
            f.write('guesses: %5d\n' % guesses)
            f.write('Actual Defects   : %4d\n' % self._results['actual_defects'] )
            f.write('Predicted Defects: %4d\n' % self._results['correctly_predicted_defects'] )
            f.write('Missed    Defects: %4d\n' % self._results['missed_defects'] )

        # self.summary = summaries
        return rgb

    def compute_all_as_image_with_defects(self, img, defects):
        results = np.zeros_like(img)
        summaries = {}
        w, h = img.shape
        rgb = np.zeros([img.shape[0],img.shape[1],3])
        # rgb[:,:,1] = img.astype('uint8')
        # rgb[:,:,2] = img.astype('uint8')
        # rgb[:,:,0] = img.astype('uint8')
        
        # for y in range(0, h, self.region_size[1]):
        #     for x in range(0, w, self.region_size[0]):
        for y in range(0, h, self.output_size[1]):
            for x in range(0, w, self.output_size[0]):

                center_x = x+int(self.region_size[0]//2)
                center_y = y+int(self.region_size[1]//2)
                hw_out = int(self.output_size[0]//2)
                hh_out = int(self.output_size[1]//2)
                start_x = center_x - hw_out
                start_y = center_y - hh_out
                end_x = center_x + hw_out
                end_y = center_y + hh_out
                s_start_x = int(self.region_size[0]/2) - hw_out
                s_start_y = int(self.region_size[1]/2) - hh_out
                s_end_x = int(self.region_size[0]/2) + hw_out
                s_end_y = int(self.region_size[1]/2) + hh_out

                has_defect = False
                for polygon in defects:
                    for defect_y, defect_x in polygon:
                        if start_x <= defect_x <= end_x and start_y <= defect_y <= end_y:
                            has_defect = True
                            break
                    if has_defect:
                        break

                result = self.compute_at(img, x, y)
                summary = self.summarize_result(result)
                if summary not in summaries:
                    summaries[summary] = {
                        'defect_count': 0,
                        'normal_count': 0,
                    }
                summary_type = 'normal_count'
                if has_defect:
                    summary_type = 'defect_count'
                summaries[summary][summary_type] += 1

                try:
                    # Color defects red, everthing else white
                    if has_defect:
                        rgb[start_y:end_y, start_x:end_x, 2] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                    else:
                        rgb[start_y:end_y, start_x:end_x, 0] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                        rgb[start_y:end_y, start_x:end_x, 1] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                        rgb[start_y:end_y, start_x:end_x, 2] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                except ValueError:
                    # Forget the ones at the end that dont fit
                    #print('%d, %d' % (x, y))
                    pass
        self.summary = summaries
        return rgb

    def compute_all_as_image(self, img):
        results = np.zeros_like(img)
        w, h = img.shape
        # for y in range(0, h, self.region_size[1]):
        #     for x in range(0, w, self.region_size[0]):
        for y in range(0, h, self.output_size[1]):
            for x in range(0, w, self.output_size[0]):
                result = self.compute_at(img, x, y)
                center_x = x+int(self.region_size[0]//2)
                center_y = y+int(self.region_size[1]//2)
                hw_out = int(self.output_size[0]//2)
                hh_out = int(self.output_size[1]//2)
                start_x = center_x - hw_out
                start_y = center_y - hh_out
                end_x = center_x + hw_out
                end_y = center_y + hh_out
                s_start_x = int(self.region_size[0]/2) - hw_out
                s_start_y = int(self.region_size[1]/2) - hh_out
                s_end_x = int(self.region_size[0]/2) + hw_out
                s_end_y = int(self.region_size[1]/2) + hh_out
                try:
                    results[start_y:end_y, start_x:end_x] = result[s_start_y:s_end_y, s_start_x:s_end_x]
                except ValueError:
                    # Forget the ones at the end that dont fit
                    #print('%d, %d' % (x, y))
                    pass
        return results


class FourierDescriptor(RegionDescriptor):
    def __init__(self):
        RegionDescriptor.__init__(self)

    def compute_at(self, img, x, y):
        region = np.copy(img)[y:y+self.region_size[1], x:x+self.region_size[0]]
        dft = cv2.dft(np.float32(region), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        # dft_shift = dft_shift.astype('uint8')
        # treshold_level, threshold_image = cv2.threshold(magnitude_spectrum.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        treshold_level, threshold_image = cv2.threshold(magnitude_spectrum.astype('uint8'), 180, 255, cv2.THRESH_BINARY)
        # treshold_level, threshold_image = cv2.threshold(magnitude_spectrum.astype('uint8'), 175, 255, cv2.THRESH_BINARY)
        return threshold_image
        # return magnitude_spectrum.astype('uint8')
        # return cv2.threshold(dft_shift, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))


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
truth_frames = np.load(files.get('dataset/Graphene_CrSi.npy'))
truth_frames = format_data(truth_frames)
truth_defects = np.load(files.get('dataset/topo_defects.npy'), allow_pickle=True)
truth_defects = truth_defects[()]

# Choose your frame
index = 29

# Sample at various percents
# Due to the way we implemented snaking it has descrete locations
# that are all the same. So we limit the value we look at to the
# unique ones. Also note:
# 0% makes no sense to rebuild.
# 1% seems to fail for gaussian reconstruction.
# targets = list(range(2,14)) + [16,18,25,33,50]
# targets = [12, 16,18,25,33,50]
# targets = [16,18,25]

# targets = [12, 13, 16, 18, 25]
# test_images = [29, 85, 3]
# targets = [18, 25]
# test_images = [29, 85, 3]
targets = [5,10,16,33,50]
test_images = [29]
for index in test_images:
    for i in targets:
        sampler = SnakeSampler(float(i)/100.0)
        # sampler = SnakeSampler(12.0/100.0)
        s_name = 'snake'
        rebuilder = GaussianRebuilder()
        r_name = 'gaussian'
        # rebuilder = FourierRebuilder()
        # r_name = 'fourier'
        marker = CrossDefectMarker()
        enhancer = LocalContrastEnhancer()

        start_time = time.time()
        img = np.copy(truth_frames[index])
        img = sampler.sample(img)
        img = rebuilder.rebuild(img)
        img = enhancer.enhance(img)
        # Fourier measurement of things:
        descriptor = FourierDescriptor()
        descriptor.output_size = (16,16)
        with open('results_%03d_descriptor.json' % i) as f:
            descriptor.load_summary(json.loads(f.read()))
        # description = descriptor.compute_at(img, 0, 0)

        with open('test_summary', 'a') as f:
            f.write('Results for index:%03d percent:%03d ' % (index, i))
        description = descriptor.classify_all_as_image_with_defects(img, truth_defects[index])
        with open('test_summary', 'a') as f:
            f.write('\n\n')

        end_time = time.time()
        print('%4.2f' % (end_time - start_time))
        marker = CrossDefectMarker()
        cv2.imwrite('results_%03d_%03d_classification_revisit.png' % (index, i), marker.mark(description, truth_defects[index]))
        # cv2.imwrite('results_%03d_%03d_classification_revisit.png' % (index, i), description)

