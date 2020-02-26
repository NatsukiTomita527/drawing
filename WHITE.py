import csv
import cv2

im = cv2.imread("smile_test.png")
h, w, _ = im.shape
SIZE_x = w
SIZE_y = h


class White:
    def __init__(self):
        xy_black = []
        with open('xy_smile.csv') as f:
            reader = csv.reader(f)
            l = [row for row in reader]
            xy_black += ([[int(v) for v in row] for row in l])

        def __init__(self):
            xy_white_true = []
            for i in range(w):
                for j in range(h):

                    # if the pixel is black, create new `food` and add this food to the list
                    if [i, j] not in xy_black:
                        xy_white_true.append([i, j])

            for i in range(len(xy_white_true)):
                self.x[i] = xy_white_true[0][i]
                self.y[i] = xy_white_true[1][i]