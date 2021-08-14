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

# Save frames 

def save_frame(frame, index, color_map):
    plt.imsave("viz/%s/frame%03d.png" % (color_map.lower(), index), frame, cmap=color_map)
    print("Saving: viz/%s/frame%03d.png" % (color_map.lower(), index))

def draw_defects_as_sequence(frame, defects):
    rgb = np.zeros([frame.shape[0],frame.shape[1],3])
    rgb[:,:,1] = frame.astype('uint8')
    rgb[:,:,2] = frame.astype('uint8')
    rgb[:,:,0] = frame.astype('uint8')
    marker_color = (0,0,255)
    line_thickness = 1
    marker_size = 2
    for polygon in defects:
        px, py = polygon[0]
        for x, y in polygon:
            cv2.line(rgb, (int(py),int(px)), (int(y),int(x)), marker_color, line_thickness)
            px, py = x, y
    return rgb


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


frames = np.load('./dataset/SMC_data_challenge/SMC_data_challenge/Graphene_CrSi.npy')
frames = format_data(frames)
defects = np.load('./dataset/SMC_data_challenge/SMC_data_challenge/topo_defects.npy', allow_pickle=True)
defects = defects[()]
for index, frame in enumerate(frames):
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # color maps used: hsv gray Paired PiYG
    # save_frame(frame, index, 'PiYG')
    rgb = draw_defects_as_cross(frame, defects[index])
    cv2.imwrite('viz/marked/cross/frame%03d.png' % index, rgb)
    print('Saving... viz/marked/cross/frame%03d.png' % index)
