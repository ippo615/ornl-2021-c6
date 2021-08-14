import matplotlib.pyplot as plt
import numpy as np
import cv2


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


frames = np.load('./dataset/SMC_data_challenge/SMC_data_challenge/Graphene_CrSi.npy')
frame = frames[0]
scanner = FakeScanner()
scanner.ground_truth = frame

result = np.zeros_like(frame)
cols, rows = frame.shape
for y in range(cols):
    for x in range(rows):
        scanner.move_to(x, y)
        result[y][x] = scanner.scan()

plt.imsave("viz/scan.png", result, cmap='gray')
print('Scanned %d points in %f seconds with %d moves' % (
    scanner.scans,
    scanner.opertating_time,
    scanner.moves
))


