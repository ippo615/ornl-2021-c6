from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.core.numeric import Inf

def format_data(frames):
    # The data coming in seems to be in the range 0.0 to 1.0
    # We convert it to 0..255 unsigned 8-bit integers so will
    # be saved as a black/white image.
    min_ = frames.min()
    max_ = frames.max()
    return ((frames - min_) * (1.0/(max_ - min_) * 255.0)).astype('uint8')

# Save frames 

def save_frame(frame, index, color_map):
    plt.imsave("viz/%s/frame%03d.png" % (color_map.lower(), index), frame, cmap=color_map)
    print("Saving: viz/%s/frame%03d.png" % (color_map.lower(), index))


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


def aabbs_from_defects(defects):
    aabbs = []
    for polygon in defects:
        min_y, min_x = np.amin(polygon,axis=0)
        max_y, max_x = np.amax(polygon,axis=0)
        aabbs.append([
            [min_x, min_y],
            [max_x, max_y]
        ])
    return aabbs


def draw_aabbs(rgb, aabbs):
    marker_color = (0,255,0)
    line_thickness = 1
    for aabb in aabbs:
        x_min = int(aabb[0][0])
        y_min = int(aabb[0][1])
        x_max = int(aabb[1][0])
        y_max = int(aabb[1][1])
        cv2.line(rgb, (x_min, y_min), (x_min, y_max), marker_color, line_thickness)
        cv2.line(rgb, (x_min, y_max), (x_max, y_max), marker_color, line_thickness)
        cv2.line(rgb, (x_max, y_max), (x_max, y_min), marker_color, line_thickness)
        cv2.line(rgb, (x_max, y_min), (x_min, y_min), marker_color, line_thickness)
    return rgb

def draw_scan_lines(rgb, scan_lines):
    marker_color = (255,0,0)
    line_thickness = 1
    for y  in scan_lines:
        for x_min, x_max in scan_lines[y]:
            cv2.line(rgb, (x_min, y), (x_max, y), marker_color, line_thickness)
    return rgb

def slice_aabbs(aabbs):
    # AABBs should be of the form:
    # [[min_x, min_y], [max_x, max_y]]
    # And `aabbs` should be a list of them so:
    # [ [[x1,y1],[x2,y2]], [[x1,y1],[x2,y2]], ... ]
    slices = defaultdict(list)
    for aabb in aabbs:
        x_min = int(aabb[0][0])
        y_min = int(aabb[0][1])
        x_max = int(aabb[1][0])
        y_max = int(aabb[1][1])
        # Alternatively we could get an aabb of an arbitrary polygon
        # min_x, min_y = np.amin(polygon,axis=0)
        # max_x, max_y = np.amax(polygon,axis=0)
        for y in range(y_min, y_max):
            slices[y].append( [x_min, x_max] )

    # We now have slices for every y position which have a list of
    # segments for x_min, x_max. We should combine them to remove
    # overlaps.
    
    for y in slices:
        # Sorting will pull the starting positions in ascending order
        cleaned = []
        current_stop = -Inf
        for start, stop in sorted(slices[y]):
            if start > current_stop:
                cleaned.append([start,stop])
                current_stop = stop
            # Segments overlap
            else:
                # Start remains the same but we either need to expand
                # this segment or we completely contain it:
                current_stop = max(current_stop, stop)
                cleaned[-1][1] = current_stop

        slices[y] = cleaned

    return slices

def filter_slices(slices, y_values):
    return {y: slices[y] for y in y_values}

frames = np.load('./dataset/SMC_data_challenge/SMC_data_challenge/Graphene_CrSi.npy')
frames = format_data(frames)
defects = np.load('./dataset/SMC_data_challenge/SMC_data_challenge/topo_defects.npy', allow_pickle=True)
defects = defects[()]
rows, cols = frames[0].shape
for index, frame in enumerate(frames):
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # color maps used: hsv gray Paired PiYG
    # save_frame(frame, index, 'PiYG')
    aabbs = aabbs_from_defects(defects[index])
    dense_scan_lines = slice_aabbs(aabbs)
    scan_lines = filter_slices(dense_scan_lines, range(0,rows,4))
    rgb = draw_defects_as_cross(frame, defects[index])
    rgb = draw_scan_lines(rgb, scan_lines)
    rgb = draw_aabbs(rgb, aabbs_from_defects(defects[index]))
    cv2.imwrite('viz/marked/07_slices/frame%03d.png' % index, rgb)
    print('Saving... viz/marked/07_slices/frame%03d.png' % index)


# print(scanner.measurements)
