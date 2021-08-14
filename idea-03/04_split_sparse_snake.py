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
    downsampled = fill_nan_with(sample_by_snaking(frame, 0.25), 0.0)
    # DEBUGGING
    # out = draw_defects_as_cross(downsampled, defects)
    # shapes = np.zeros_like(out)

    defect_xys = []
    for polygon in defects:
        for x, y in polygon:
            defect_xys.append((x,y))

    w, h = frame.shape[0], frame.shape[1]
    subw, subh = region_size
    xcount = int(w//subw)
    ycount = int(h//subh)
    for xidx in range(xcount):
        for yidx in range(ycount):
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
                cv2.imwrite(base_filename % ('defect', xs, ys), downsampled[ys:ye, xs:xe])
                # DEBUGGING
                # cv2.rectangle(shapes, (ys, xs), (ye, xe), (0, 255, 0), cv2.FILLED)
            else:
                cv2.imwrite(base_filename % ('normal', xs, ys), downsampled[ys:ye, xs:xe])

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



frames = np.load('../dataset/Graphene_CrSi.npy')
frames = format_data(frames)
defects = np.load('../dataset/topo_defects.npy', allow_pickle=True)
defects = defects[()]
for index, frame in enumerate(frames):
    # '{folder}/{defect/normal}/{index}_{x}_{y}_{method}.png'
    base_filename = '%s/%%s/%03d_%%04d_%%04d_%s.png' % (
        '04_split_snake', # folder
        index,
        'sparse' # method
    )
    split_image_with_defects(frame, defects[index], (64, 64), base_filename)
    print('Ran frame %03d' % index)
