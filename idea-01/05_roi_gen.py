import matplotlib.pyplot as plt
import numpy as np
import cv2

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
        min_x, min_y = np.amin(polygon,axis=0)
        max_x, max_y = np.amax(polygon,axis=0)
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
        cv2.line(rgb, (y_min, x_min), (y_min, x_max), marker_color, line_thickness)
        cv2.line(rgb, (y_min, x_max), (y_max, x_max), marker_color, line_thickness)
        cv2.line(rgb, (y_max, x_max), (y_max, x_min), marker_color, line_thickness)
        cv2.line(rgb, (y_max, x_min), (y_min, x_min), marker_color, line_thickness)
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
    rgb = draw_aabbs(rgb, aabbs_from_defects(defects[index]))
    cv2.imwrite('viz/marked/05_roi/frame%03d.png' % index, rgb)
    print('Saving... viz/marked/05_roi/frame%03d.png' % index)



# print(scanner.measurements)
