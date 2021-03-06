import cv2
import math
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective


class MathManager:
    def getPointToMM(self, img, point):
        pixelsPerMetric = self.getPixelPerMetrix(img)
        return (point[0] / pixelsPerMetric) * 10, (point[1] / pixelsPerMetric) * 10

    @staticmethod
    def getAngleFunc(a, b, h):
        angleAlpha = math.atan(a/b)
        s = h * math.tan(angleAlpha)
        return s

    @staticmethod
    def getPiceMidpoint(ctr):
        M = cv2.moments(ctr)
        if M["m10"] > 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            return x, y

    @staticmethod
    def getAngleBetweenLines(midpoint, p1, p2):
        angle1 = math.atan2(midpoint[1] - p1[1], midpoint[0] - p1[0])
        angle2 = math.atan2(midpoint[1] - p2[1], midpoint[0] - p2[0])
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
        x = ((p[0] - midpoint[0]) * math.cos(a)) - ((p[1] - midpoint[1]) * math.sin(a)) + midpoint[0]
        y = ((p[0] - midpoint[0]) * math.sin(a)) + ((p[1] - midpoint[1]) * math.cos(a)) + midpoint[1]
        return x, y

    @staticmethod
    def midpoint(p1, p2):
        return (p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5

    def getPixelPerMetrix(self, img):
        areaofInterest_width_px = img.shape[1]
        areaofInterest_width_cm = 36.55  # cm
        return areaofInterest_width_px / areaofInterest_width_cm

    @staticmethod
    def getMinAreaBoxPoint(ctr):
        box = cv2.minAreaRect(ctr)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        return box

    def getPiceDimension(self, ctr, img):
        pixelsPerMetric = self.getPixelPerMetrix(img)
        box = self.getMinAreaBoxPoint(ctr)
        (tl, tr, br, bl) = box

        cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
        (tltrX, tltrY) = self.midpoint(tl, tr)
        (blbrX, blbrY) = self.midpoint(bl, br)

        (tlblX, tlblY) = self.midpoint(tl, bl)
        (trbrX, trbrY) = self.midpoint(tr, br)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # draw lines between the midpoints
        cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 1)
        cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 1)
        cv2.putText(img, "a{:.2f}cm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 0), 1)
        cv2.putText(img, "b{:.2f}cm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 0), 1)

        return dimA, dimB

    @staticmethod
    def getFocalLength():
        focalLength = 1435.4520547945206
        return focalLength

    @staticmethod
    def calcFocalLength(KNOWN_DISTANCE, areaofInterest_width_px=0.0, areaofInterest_width_cm=0.0):
        return (areaofInterest_width_px * KNOWN_DISTANCE) / areaofInterest_width_cm

    def movePointByDistance(self, p, s, a, img):
        new_p = int(math.sqrt(math.pow((s + p[0]), 2))), int(math.sqrt(math.pow((s + p[1]), 2)))
        cv2.circle(img, new_p, 5, (0, 0, 0), -1)
        return self.rotatePoint(new_p, p, a)

    def getMinAreaBoxRotation(self, box, id):
        (tl, tr, br, bl) = box
        dist_bl_br = self.getPointDistance(bl, br)
        dist_bl_tl = self.getPointDistance(bl, tl)

        if dist_bl_br > dist_bl_tl:
            if id == 0:  # Elefant
                return self.getAngleBetweenLines(bl, br, (br[0], bl[1])) + 11
            if id == 1:  # Frosch
                return self.getAngleBetweenLines(bl, br, (br[0], bl[1]))
            if id == 2:  # Lowe
                return self.getAngleBetweenLines(bl, tl, (br[0], bl[1])) + 5
            if id == 3:  # Schmetterling
                return 0
            if id == 4:  # Sonne
                return self.getAngleBetweenLines(bl, br, (br[0], bl[1]))
            if id == 5:  # Vogel
                return self.getAngleBetweenLines(bl, br, (br[0], bl[1])) + 45

        else:
            if id == 0:  # Elefant
                return self.getAngleBetweenLines(bl, tl, (br[0], bl[1])) + 11
            if id == 1:  # Frosch
                return self.getAngleBetweenLines(bl, br, (br[0], bl[1]))
            if id == 2:  # Lowe
                return self.getAngleBetweenLines(bl, br, (br[0], bl[1])) + 7
            if id == 3:  # Schmetterling
                return 0
            if id == 4:  # Sonne
                return self.getAngleBetweenLines(bl, tl, (br[0], bl[1]))
            if id == 5:  # Vogel
                return self.getAngleBetweenLines(bl, br, (br[0], bl[1])) - 45

    @staticmethod
    def findIntersection(p1, p2, p3, p4):
        px = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (p3[0] * p4[1] - p3[1] * p4[0])) / \
             ((p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]))
        py = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] * p4[1] - p3[1] * p4[0])) / \
             ((p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]))
        return px, py

    @staticmethod
    def getCameraDistance(knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / perWidth

    def rotateContur(self, ctr, angle, midpoint):
        ctr_rotated = list()
        for c in ctr:
            ctr_rotated.append(self.rotatePoint(c, midpoint, angle))
        return ctr_rotated

    def getTransformedContour(self, midpoint, normedcorners, normdmidpoint):
        ctr = normedcorners
        for i, corner in enumerate(normedcorners):
            ctr[i] = [(midpoint[0] - normdmidpoint[0]) + corner[0], (midpoint[1] - normdmidpoint[1]) + corner[1]]
        return ctr

    @staticmethod
    def getRavelCorner(corners):
        print(corners)

        cornerlist = list()
        for corner in corners:
            x, y = corner.ravel()
            cornerlist.append((x,y))
        return cornerlist

    def getPiceRotation(self, ctr, id, img):
        box = self.getMinAreaBoxPoint(ctr)
        (tl, tr, br, bl) = box

        boxrotation = self.getMinAreaBoxRotation(box, id)
        print(boxrotation)
        return boxrotation

    @staticmethod
    def getExtremePoints(c):
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        return (extLeft, extRight, extTop, extBot)