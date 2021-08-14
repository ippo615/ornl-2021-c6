import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.lib.polynomial import poly

def format_data(frames):
    # The data coming in seems to be in the range 0.0 to 1.0
    # We convert it to 0..255 unsigned 8-bit integers so will
    # be saved as a black/white image.
    min_ = frames.min()
    max_ = frames.max()
    return ((frames - min_) * (1.0/(max_ - min_) * 255.0)).astype('uint8')

def draw_defects_as_cross(frame, defects):
    rgb = np.zeros([frame.shape[0],frame.shape[1],3])
    rgb[:,:,1] = frame.astype('uint8')
    rgb[:,:,2] = frame.astype('uint8')
    rgb[:,:,0] = frame.astype('uint8')
    marker_color = (0,0,255)
    line_thickness = 1
    marker_size = 2
    for polygon in defects:
        for x, y in polygon:
            cv2.line(rgb, (int(y-marker_size),int(x)), (int(y+marker_size),int(x)), marker_color, line_thickness)
            cv2.line(rgb, (int(y),int(x-marker_size)), (int(y),int(x+marker_size)), marker_color, line_thickness)
    return rgb


def split_image_with_defects(frame, defects, region_size, base_filename):
    # downsampled = fill_nan_with(sample_by_snaking(frame, 0.25), 0.0)
    downsampled = frame
    # DEBUGGING
    # out = draw_defects_as_cross(downsampled, defects)
    # shapes = np.zeros_like(out)

    defect_xys = []
    for polygon in defects:
        for x, y in polygon:
            defect_xys.append((x,y))

    # lazy way of ignoring the large white blob
    # as a list of (y, x) pairs
    ignore_regions = list(zip([ 0]*16,range(4,16)))
    ignore_regions+= list(zip([ 1]*16,range(4,16)))
    ignore_regions+= list(zip([ 2]*16,range(5,16)))
    ignore_regions+= list(zip([ 3]*16,range(7,16)))
    ignore_regions+= list(zip([ 4]*16,range(8,16)))
    ignore_regions+= list(zip([ 5]*16,range(9,16)))
    ignore_regions+= list(zip([ 6]*16,range(9,16)))
    ignore_regions+= list(zip([ 7]*16,range(9,16)))
    ignore_regions+= list(zip([ 8]*16,range(8,16)))
    ignore_regions+= list(zip([ 9]*16,range(10,16)))
    ignore_regions+= list(zip([10]*16,range(10,16)))
    ignore_regions+= list(zip([11]*16,range(12,16)))
    ignore_regions+= list(zip([12]*16,range(12,16)))
    ignore_regions+= list(zip([13]*16,range(12,16)))
    ignore_regions+= list(zip([14]*16,range(12,16)))
    ignore_regions+= list(zip([15]*16,range(12,16)))

    w, h = frame.shape[0], frame.shape[1]
    subw, subh = region_size
    xcount = int(w//subw)
    ycount = int(h//subh)
    for xidx in range(xcount):
        for yidx in range(ycount):
            if (xidx, yidx) in ignore_regions:
                continue
            xs = xidx * subw
            ys = yidx * subh
            xe = xs + subw
            ye = ys + subh
            has_defect = False
            for x,y in defect_xys:
                if xs <= x < xe and ys <= y < ye:
                    has_defect = True
                    break
            if has_defect:
                cv2.imwrite(base_filename % ('defect', xs, ys), downsampled[xs:xe, ys:ye])
                # DEBUGGING
                # cv2.rectangle(shapes, (ys, xs), (ye, xe), (0, 255, 0), cv2.FILLED)
            else:
                cv2.imwrite(base_filename % ('normal', xs, ys), downsampled[xs:xe, ys:ye])

    # for x, y in defect_xys:
    #     xs = int(x-subw//2)
    #     ys = int(y-subh//2)
    #     xe = xs + subw
    #     ye = ys + subh
    #     if 0 <= xs < xe < w and 0 <= ys < ye < h:
    #         cv2.imwrite(base_filename % ('defect', xs, ys), downsampled[xs:xe, ys:ye])


    # DEBUGGING
    # Generate output by blending image with shapes image, using the shapes
    # images also as mask to limit the blending to those parts
    # alpha = 0.5
    # mask = shapes.astype(bool)
    # out[mask] = cv2.addWeighted(out, alpha, shapes, 1 - alpha, 0)[mask]
    # cv2.imwrite('debug.png', out)


def fill_nan_with(arr, value):
    mask = np.isnan(arr)
    arr[mask] = value
    return arr


def sample_by_snaking( frame, max_percent ):
    sparse_image = np.zeros_like(frame)
    sparse_image.fill(np.nan)
    w, h = frame.shape
    # Horizontal scan lines
    # percent_scanned = pixels_scanned / (w*h)
    # pixels_scanned = percent_scanned * (w*h)
    # pixels_scanned = w*scan_lines+h
    # percent_scanned * w * h = w*scan_lines+h
    # (percent_scanned * w * h - h) / w = scan_lines

    n_scan_lines = int((max_percent * w * h - h) / w)
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

def local_enhancements(frame):
    original_type = frame.dtype
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    frame = clahe.apply(format_data(frame))
    return frame.astype(original_type) / frame.max()

def process_image(frame, sample_percent, fmin, fmax, defects):
    # img = format_data(local_enhancements(np.copy(frame)))
    # return img
    # return draw_defects_as_cross(img, defects)
    return format_data(filter_frequencies(local_enhancements(np.copy(frame)), fmin=fmin, fmax=fmax))
    # img = np.copy(local_enhancements(frame))
    # sparse_image = fill_nan_with(sample_by_snaking(img, sample_percent), 0.0)
    # return filter_frequencies(sparse_image, fmin, fmax)

frames = np.load('../dataset/Graphene_CrSi.npy')
frames = format_data(frames)
defects = np.load('../dataset/topo_defects.npy', allow_pickle=True)
defects = defects[()]
for index, frame in enumerate(frames):
    # '{folder}/{defect/normal}/{index}_{x}_{y}_{method}.png'
    base_filename = '%s/%%s/%03d_%%04d_%%04d_%s.png' % (
        '08_fourier', # folder
        index,
        'full' # method
    )
    img = process_image(frame, 1.0, 0, 100, defects[index])
    split_image_with_defects(img, defects[index], (64, 64), base_filename)
    # split_image_with_defects(frame, defects[index], (64, 64), base_filename)
    print('Ran frame %03d' % index)

# Splitting on defect centers and normal grid
# 50474 defects - 88%
#  6671 normal  - 12%

# Splitting purely on grid
# 7629 defects - 53%
# 6671 normal - 47%

