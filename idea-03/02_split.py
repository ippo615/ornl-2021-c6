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

def color_regions_with_defects(frame, defects, region_size):
    # assumes frame is already a rgb image with marks
    # Initialize blank mask image of same dimensions for drawing the shapes
    shapes = np.zeros_like(frame)

    defect_xys = []
    for polygon in defects:
        for x, y in polygon:
            defect_xys.append((x,y))

    w, h, depth = frame.shape
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
                # cv2.imwrite(base_filename % ('defect', xidx*subw, yidx*subh), )
                cv2.rectangle(shapes, (ys, xs), (ye, xe), (0, 255, 0), cv2.FILLED)

    # Generate output by blending image with shapes image, using the shapes
    # images also as mask to limit the blending to those parts
    alpha = 0.5
    out = frame.copy()
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

    return out


def split_image_with_defects(frame, defects, region_size, base_filename):
    # assumes frame is already a rgb image with marks
    # Initialize blank mask image of same dimensions for drawing the shapes
    shapes = np.zeros_like(frame)

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
                cv2.imwrite(base_filename % ('defect', xs, ys), frame[ys:ye, xs:xe])
            else:
                cv2.imwrite(base_filename % ('normal', xs, ys), frame[ys:ye, xs:xe])


frames = np.load('../dataset/Graphene_CrSi.npy')
frames = format_data(frames)
defects = np.load('../dataset/topo_defects.npy', allow_pickle=True)
defects = defects[()]
for index, frame in enumerate(frames):
    # '{folder}/{defect/normal}/{index}_{x}_{y}_{method}.png'
    base_filename = '%s/%%s/%03d_%%04d_%%04d_%s.png' % (
        '02_split_grid', # folder
        index,
        'original' # method
    )
    split_image_with_defects(frame, defects[index], (64, 64), base_filename)
    print('Ran frame %03d' % index)
