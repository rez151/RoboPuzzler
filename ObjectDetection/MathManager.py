import cv2
import math
import numpy as np
import imutils


class MathManager:
    def midPointToCm(self, midpoint):
        dpi = 150
        conversion_factor = 2.54
        x = round(((midpoint[0] * conversion_factor) / dpi)*10, 4)
        y = round(((midpoint[1] * conversion_factor) / dpi)*10, 4)
        return x, y

    @staticmethod
    def getMidpoint(ctr):
        M = cv2.moments(ctr)
        if M["m10"] > 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            return x, y

    def normedMaxPosition(self, midpoint, ctr):
        ox, oy, omx, omy = self.getOrientationPoint(ctr)
        mx, my = midpoint
        ox = (mx - omx) + ox
        oy = (my - omy) + oy
        return ox, oy

    @staticmethod
    def getRotation(midpoint, maxpoint, normedmaxpoint):
        angle1 = math.atan2(midpoint[1] - maxpoint[1], midpoint[0] - maxpoint[0])
        angle2 = math.atan2(midpoint[1] - normedmaxpoint[1], midpoint[0] - normedmaxpoint[0])
        result = (angle2 - angle1) * 180 / math.pi
        if result > 180:
            result -= 360
        return result

    @staticmethod
    def getPointDistance(p1, p2):
        return math.sqrt(math.pow((p2[1] - p1[1]), 2) + math.pow((p2[0] - p1[0]), 2))

    def getPointMaxDistance(self, midpoint, p):
        p = np.unique(p, axis=0)
        maxdist = 0
        maxpoint = (0, 0)
        for p1 in p:
            for p2 in p1:
                if maxdist < self.getPointDistance(midpoint, p2):
                    maxdist = self.getPointDistance(midpoint, p2)
                    maxpoint = p2
        return maxpoint[0], maxpoint[1]

    def getPointMinDistance(self, midpoint, p):
        p = np.unique(p, axis=0)
        mindist = 0
        minpoint = {0, 0}
        for p1 in p:
            for p2 in p1:
                if mindist > self.getPointDistance(midpoint, p2):
                    mindist = self.getPointDistance(midpoint, p2)
                    minpoint = p2

        return minpoint[0], minpoint[1]

    def getRadius(self, midpoint, p):
        return self.getPointDistance(midpoint, p)

    @staticmethod
    def rotatePoint(p, midpoint, a):
        x0, y0 = midpoint
        x1, y1 = p
        x2 = ((x1 - x0) * math.cos(a)) - ((y1 - y0) * math.sin(a)) + x0
        y2 = ((x1 - x0) * math.sin(a)) + ((y1 - y0) * math.cos(a)) + y0
        return x2, y2

    def getOrientationPoint(self, ctr):
        mx, my = self.getMidpoint(ctr)
        x, y = self.getPointMaxDistance((mx, my), ctr)
        return x, y, mx, my
