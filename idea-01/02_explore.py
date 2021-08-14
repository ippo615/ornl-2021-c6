import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.lib.polynomial import poly

def stats_for_frame(frame, index, print_func):
    print_func('Frame: %03d' % index)
    print_func('  max: %f' % frame.max())
    print_func('  min: %f' % frame.min())

frames = np.load('./dataset/SMC_data_challenge/SMC_data_challenge/Graphene_CrSi.npy')
defects = np.load('./dataset/SMC_data_challenge/SMC_data_challenge/topo_defects.npy', allow_pickle=True)
defects = defects[()]
for index, frame in enumerate(frames):
    stats_for_frame(frame, index, print)
