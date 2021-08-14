import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.core.fromnumeric import argmax
from numpy.core.numeric import Inf, zeros_like

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
            cv2.line(rgb, (int(x-marker_size),int(y)), (int(x+marker_size),int(y)), marker_color, line_thickness)
            cv2.line(rgb, (int(x),int(y-marker_size)), (int(x),int(y+marker_size)), marker_color, line_thickness)
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

def hulls_from_defects(defects):
    hulls = []
    for polygon in defects:
        # print(polygon)
        # print(np.array([[1,2], [10,10], [15,20]]))
        # print('WTF??')
        #hulls.append(cv2.convexHull(polygon.astype('int')))
        #raise RuntimeError()
        hulls.append(cv2.convexHull(polygon.astype('int')))
        # hulls.append(cv2.convexHull(np.array([[1,2], [10,10], [15,20]])))
    return hulls


def draw_hulls(rgb, hulls):
    marker_color = (0,255,0)
    line_thickness = 1
    for hull in hulls:
        cv2.fillPoly(rgb, [hull], marker_color)
        # xp, yp = hull[0][0]
        # for point in hull:
        #     x, y = point[0]
        #     cv2.line(rgb, (yp, xp), (y, x), marker_color, line_thickness)
        #     xp, yp = x, y
        # x, y = hull[0][0]
        # cv2.line(rgb, (yp, xp), (y, x), marker_color, line_thickness)
    return rgb



class FakeScanner:
    def __init__(self):
        self.ground_truth = None
        self.opertating_time = 0.0
        self.moves = 0
        self.scans = 0
        self.x = 0
        self.y = 0

    def move_to(self, x, y):
        self.x = x
        self.y = y
        # todo: increase operating time as a function of distance
        self.opertating_time += 0.0
        self.moves += 1

    def scan(self):
        self.opertating_time += 32e-6  # seconds (ie 32 microseconds)
        self.scans += 1
        return self.ground_truth[self.y][self.x]


class MeanderingScanner(FakeScanner):
    def __init__(self):
        FakeScanner.__init__(self)
        self.x_dir = 0
        self.y_dir = 0
        self.targets = None
        self.visited = None
        self._target = [0,0]

    def initialize(self, ground_truth, defects):
        self.ground_truth = ground_truth

        # Convert the defects to a convex hull and draw
        # them on the targets image
        rows, cols = self.ground_truth.shape
        self.targets = np.zeros((rows,cols))
        for polygon in defects:
            hull = cv2.convexHull(polygon.astype('int'))
            cv2.fillPoly(self.targets, [hull], 255)

        self.visited = np.zeros_like(self.targets)

    # simple ruleset:
    # Let's consider 2 scenarios:
    #   we are either in-region
    #   or out-of-region
    # If we're out-of-region we will pick a target to head toward.
    # We will continue to move in the maximal x/y delta to move.
    # If we are in region we will snake through the entire thing.

    def _step_find_target(self):
        # find a new target x,y
        # TODO
        pass

    def _step_toward_target(self):

        if not self._target:
            self._step_find_target()

        dx = self._target[0] - self.x
        dy = self._target[1] - self.y

        # We reached the goal
        if dx == 0 and dy == 0:
            self._target = []

        # determine if we should go up/down/left/right
        dir_x = 0
        dir_y = 0
        if abs(dx) > abs(dy):
            dir_x = 1 if dx > 0 else -1
        else:
            dir_y = 1 if dy > 0 else -1
        
        # TODO

    def _step_thru_target(self):
        nx = self.x + self.x_dir
        ny = self.y + self.y_dir
        is_next_on_target = self.targets[nx][ny] != 0
        if is_next_on_target:
            self.move_to(nx,ny)
            return
        
        # Not on target -> TURN!...maybe
        # TODO


    def step(self):
        is_on_target = self.targets[self.x][self.y] != 0
        if is_on_target:
            self._step_thru_target()
        else:
            self._step_toward_target()
        
        nx = self.x + self.x_dir
        ny = self.y + self.y_dir
        

class ManhattanScanner(FakeScanner):
    def __init__(self):
        FakeScanner.__init__(self)
        self.x_dir = 0
        self.y_dir = 0
        self.targets = None
        self.visited = None
        self._target = [0,0]

    def initialize(self, ground_truth, defects):
        self.ground_truth = ground_truth

        # Convert the defects to a convex hull and draw
        # them on the targets image
        rows, cols = self.ground_truth.shape
        self.targets = np.zeros((rows,cols))
        for polygon in defects:
            hull = cv2.convexHull(polygon.astype('int'))
            cv2.fillPoly(self.targets, [hull], 255)

        self.visited = np.zeros_like(self.targets)

    def score_next_location(self, point):
        nx, ny = point

        # Avoid re-visiting points
        is_next_visited = self.targets[nx][ny] != 0
        if is_next_visited:
            return -1

        dx, dy = nx-self.x, ny-self.y
        is_same_direction = (dx == self.x_dir and dy == self.y_dir)
        is_on_target = self.targets[self.x][self.y] != 0
        is_next_on_target = self.targets[nx][ny] != 0

        # Prioritize reaching a target
        if not is_on_target:
            if is_next_on_target:
                if is_same_direction:
                    return 101
                else:
                    return 100
            else:
                return 0
                # TODO: prioritize best direction to target
                pass

        if is_on_target and is_next_on_target:
            if is_same_direction:
                return 2
            else:
                return 1

    def step(self):
        # TODO: determine when to change targets
        options = [
            [self.x+1, self.y],
            [self.x, self.y+1],
            [self.x-1, self.y],
            [self.x, self.y-1],
        ]
        scores = [ self.score_next_location(o) for o in options ]
        best_index = np.argmax(scores)
        return options[best_index]


frames = np.load('./dataset/SMC_data_challenge/SMC_data_challenge/Graphene_CrSi.npy')
frames = format_data(frames)
defects = np.load('./dataset/SMC_data_challenge/SMC_data_challenge/topo_defects.npy', allow_pickle=True)
defects = defects[()]
# Make them (x,y) pairs - by default they seem to be (y,x) pairs
for index in defects:
    defect = defects[index]
    for group in defect:
        group[:,[0, 1]] = group[:,[1, 0]]
rows, cols = frames[0].shape
for index, frame in enumerate(frames):
    result = np.zeros_like(frame)
    start = time.time()
    scanner = ManhattanScanner()
    scanner.initialize(frame, defects[index])
    for i in range(1000):
        x,y = scanner.step()
        scanner.move_to(x,y)
        result[y][x] = scanner.scan()
    end = time.time()
    plt.imsave("viz/marked/09_pong/scan_%d.png" % index, result, cmap='gray')
    print('[%03d] Scanned %d points in %f microscope seconds with %d moves and %f computational seconds' % (
        index,
        scanner.scans,
        scanner.opertating_time,
        scanner.moves,
        (end-start)
    ))


# print(scanner.measurements)
