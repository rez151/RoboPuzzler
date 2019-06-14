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
        x0, y0 = midpoint
        x1, y1 = p
        x2 = ((x1 - x0) * math.cos(a)) - ((y1 - y0) * math.sin(a)) + x0
        y2 = ((x1 - x0) * math.sin(a)) + ((y1 - y0) * math.cos(a)) + y0
        return x2, y2

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
        cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 1)
        cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 1)
        cv2.putText(img, "a{:.2f}cm".format(dimA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.3, (0, 0, 0), 1)
        cv2.putText(img, "b{:.2f}cm".format(dimB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.3, (0, 0, 0), 1)

        return dimA, dimB

    @staticmethod
    def getFocalLength():
        focalLength = 1435.4520547945206
        return focalLength

    @staticmethod
    def calcFocalLength(KNOWN_DISTANCE, areaofInterest_width_px=0.0, areaofInterest_width_cm=0.0):
        return (areaofInterest_width_px * KNOWN_DISTANCE) / areaofInterest_width_cm

    def getRealWorldMap(self, img):
        table_real_x_cm = 36.5

        table_x = img.shape[1]
        table_y = img.shape[0]
        print("table_x: {}px".format(table_x))
        print("table_y: {}px".format(table_y))

        pixelsPerMetric = self.getPixelPerMetrix(img)

        pice_height = 2.5 * pixelsPerMetric
        print("pice_height: {}mm".format(pice_height / pixelsPerMetric))
        print("pice_height: {}px".format(pice_height))

        focallength =  self.getFocalLength()
        camera_height = self.getCameraDistance(table_real_x_cm, focallength, table_x)
        print("camera_height: {}cm".format(camera_height))
        camera_height = camera_height * pixelsPerMetric

        alpha = (math.atan(camera_height/(table_x / 2)))
        beta = (math.atan(camera_height/(table_y / 2)))
        print("alpha: {}°".format(math.degrees(alpha)))
        print("beta: {}°".format(math.degrees(beta)))

        pice_camera_distance = camera_height - pice_height

        table_x_transformed = int(round(pice_camera_distance / math.tan(alpha), 0))*2
        table_y_transformed = int(round(pice_camera_distance / math.tan(beta), 0))*2
        print("table_x_transformed: {}px".format(table_x_transformed))
        print("table_y_transformed: {}px".format(table_y_transformed))

        return table_x_transformed, table_y_transformed

    def movePointByDistance(self, p, s, a, img):
        new_p = int(math.sqrt(math.pow((s + p[0]), 2))), int(math.sqrt(math.pow((s + p[1]), 2)))
        cv2.circle(img, new_p, 5,(0, 0, 0), -1)
        return self.rotatePoint(new_p, p, a)

    def getMinAreaBoxRotation(self, box):
        (tl, tr, br, bl) = box
        return self.getAngleBetweenLines(br, bl, (0, br[1]))

    def getPiceRotation(self, ctr, id, img):
        box = self.getMinAreaBoxPoint(ctr)
        boxrotation = self.getMinAreaBoxRotation(box)
        dimA, dimB = self.getPiceDimension(ctr, img)
        self.getPiceOrientation(box)
        if id == 0: # Elefant
            if self.getPiceOrientation(box):
                return boxrotation + 12 +90
            else:
                return boxrotation + 12

        if id == 1: # Frosch
            return boxrotation
        if id == 2: # Lowe
            if self.getPiceOrientation(box):
                return boxrotation + 5
            else:
                return boxrotation + 5 + 90
        if id == 3: # Schmetterling
            return boxrotation
        if id == 4: # Sonne
            return boxrotation - 90
        if id == 5: # Vogel
            return boxrotation - 42

    def getPiceOrientation(self, box):
        (tl, tr, br, bl) = box
        x = self.getPointDistance(tl, bl)
        y = self.getPointDistance(tr, bl)
        if x > y:
            return True
        else:
            return False



    @staticmethod
    def getCameraDistance(knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / perWidth
