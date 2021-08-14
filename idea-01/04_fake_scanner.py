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


class BoundedScanner(FakeScanner):
    def __init__(self, width, height, step_size):
        FakeScanner.__init__(self)
        self.width = width
        self.height = height
        self.step_size = step_size
        self._has_points = True

    def add_measurement(self, value):
        pass

    def has_points(self):
        self.get_next_point()
        return self._has_points

    def get_next_point(self):
        nx = self.x+self.step_size[0]
        ny = self.y
        if nx >= self.width:
            nx = 0
            ny += self.step_size[1]
            if ny >= self.height:
                self._has_points = False
        return (nx, ny)


class BoundedThresholdScanner(FakeScanner):
    def __init__(self, width, height, step_size):
        FakeScanner.__init__(self)
        self.width = width
        self.height = height
        self.step_size = step_size
        self._has_points = True
        self.measurements = []
    
    def add_measurement(self, value):
        self.measurements.append(value)

    def has_points(self):
        self.get_next_point()
        return self._has_points

    def _get_measurement_delta(self, n):
        if not self.measurements:
            return 0.0
        if n > len(self.measurements):
            average = sum(self.measurements) / len(self.measurements)
        else:
            average = sum(self.measurements[-n:-1]) / n
        return abs(average - self.measurements[-1])

    def _last_measurement(self, n=1):
        if not self.measurements:
            return 0.0
        if n > len(self.measurements):
            average = sum(self.measurements) / len(self.measurements)
        elif n == 1:
            average = self.measurements[-1]
        else:
            average = sum(self.measurements[-n:-1]) / n
        return average

    def get_next_point(self):
        nx = self.x+self.step_size[0]
        ny = self.y
        # dm = self._get_measurement_delta(5)
        # new_line = (dm > 0.3) or (nx >= self.width)
        new_line = (self._last_measurement() > 0.4) or (nx >= self.width)
        if new_line:
            nx = 0
            ny += self.step_size[1]
            if ny >= self.height:
                self._has_points = False
        return (nx, ny)


frames = np.load('./dataset/SMC_data_challenge/SMC_data_challenge/Graphene_CrSi.npy')
frame = frames[0]
cols, rows = frame.shape
scanner = BoundedThresholdScanner(cols, rows, (4, 4))
scanner.ground_truth = frame

result = np.zeros_like(frame)
while scanner.has_points():
    (x, y) = scanner.get_next_point()
    scanner.move_to(x, y)
    result[y][x] = scanner.scan()
    scanner.add_measurement(result[y][x])



plt.imsave("viz/scan.png", result, cmap='gray')
print('Scanned %d points in %f seconds with %d moves' % (
    scanner.scans,
    scanner.opertating_time,
    scanner.moves
))
# print(scanner.measurements)
