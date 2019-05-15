import ObjectDetection.CameraManager as CameraManager
import ObjectDetection.Classifier as Classifier
from PIL import Image
import cv2
import math
import numpy as np
import imutils


class PiceManager:
    def extractPices(self, img_filtered, img_input):
        extractedPices = []
        cnts, _ = cv2.findContours(img_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        image = img_input.copy()

        for i, ctr in enumerate(sorted_ctrs):
            if (i == 0):
                pass
            else:
                # get Extracted pice
                extractPice = self.getExtractPice(img_filtered, ctr)
                extractPiceClassification  = self.getExtractPice(img_input, ctr)
                midpoint = self.getMidpoint(ctr)
                maxpoint = self.getPointMaxDistance(midpoint, ctr)
                classifierID,id =  Classifier.Classifire().Classifier(extractPice)
                normedmaxpoint = self.normedMaxPosition(midpoint, classifierID)
                rotation = self.getRotation(midpoint, maxpoint, normedmaxpoint)

                # draw line from normedpoint and local maxpoint to midpoint
                cv2.line(image,midpoint,normedmaxpoint,(255,0,0),1)
                cv2.line(image,midpoint,maxpoint,(0,255,0),1)
                # correct Rotation
                extractPice = imutils.rotate_bound(extractPice, rotation)
                # draw Contours
                cv2.drawContours(image, [ctr], 0, (0, 0, 255), 2)
                # draw Midpoint
                cv2.circle(image, midpoint, 7, (0, 255, 0), -1)
                # draw MaxPoint
                cv2.circle(image, maxpoint, 7, (0, 255, 0), -1)
                # draw Classification text
                cv2.putText(image, (str(classifierID)), midpoint, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255),2)
                # draw rotation Circles
                PiceManager().drawRotationCircle(image, midpoint, maxpoint, classifierID)
                # print progress status
                print(str(int((i * 100) / (sorted_ctrs.__len__() - 1))) + "% Done")

                extractedPices.insert(i, [i, extractPice, midpoint, str(id), rotation])
        return extractedPices, image

    def getExtractPice(self, img_filtered, ctr):
        x, y, w, h = cv2.boundingRect(ctr)
        extractPice = img_filtered[y:y + h, x:x + w]
        return extractPice


    def getContour(self, img):
        cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for i, ctr in enumerate(sorted_ctrs):
            if (i == 0):
                pass
            else:
                return ctr

    def getOrientationPoint(self, id):
        img_filtered, img_input = CameraManager.CameraManager().getImageFilebyID(id)
        normedPiceContour = self.getContour(img_filtered)
        normedPice = self.getExtractPice(img_filtered, normedPiceContour)
        mx, my = self.getMidpoint(normedPiceContour)
        x, y = self.getPointMaxDistance((mx, my), normedPiceContour)
        return x, y, mx, my

    def normedMaxPosition(self, midpoint, classifireID):
        ox, oy, omx, omy = self.getOrientationPoint(classifireID)
        mx, my = midpoint
        ox = (mx - omx) + ox
        oy = (my - omy) + oy
        return ox, oy

    def getRotation(self, midpoint, maxpoint, normedmaxpoint):
        angle1 = math.atan2(midpoint[1] - maxpoint[1], midpoint[0] - maxpoint[0])
        angle2 = math.atan2(midpoint[1] - normedmaxpoint[1], midpoint[0] - normedmaxpoint[0]);
        result = (angle2 - angle1) * 180 / math.pi;
        if (result > 180):
            result -= 360
        return result

    def getMidpoint(self, ctr):
        M = cv2.moments(ctr)
        if M["m10"] > 0:
            mX = int(M["m10"] / M["m00"])
            mY = int(M["m01"] / M["m00"])
            return mX, mY

    def getPointDistance(self, p1, p2):
        return math.sqrt(math.pow((p2[1] - p1[1]), 2) + math.pow((p2[0] - p1[0]), 2))

    def getPointMaxDistance(self, midpoint, p):
        p = np.unique(p, axis=0)
        maxdist = 0
        maxpoint = (0, 0)
        for p1 in p:
            for p2 in p1:
                if (maxdist < self.getPointDistance(midpoint, p2)):
                    maxdist = self.getPointDistance(midpoint, p2)
                    maxpoint = p2
        return maxpoint[0], maxpoint[1]

    def getPointMinDistance(self, midpoint, p):
        p = np.unique(p, axis=0)
        mindist = 0
        minpoint = {0, 0}
        for p1 in p:
            for p2 in p1:
                if (mindist > self.getPointDistance(midpoint, p2)):
                    mindist = self.getPointDistance(midpoint, p2)
                    minpoint = p2
        return minpoint[0], minpoint[1]

    def rotatePoint(self, p, midpoint, a):
        x0, y0 = midpoint
        x1, y1 = p
        x2 = ((x1 - x0) * math.cos(a)) - ((y1 - y0) * math.sin(a)) + x0;
        y2 = ((x1 - x0) * math.sin(a)) + ((y1 - y0) * math.cos(a)) + y0;
        return x2, y2

    def drawRotationCircle(self, img, midpoint, maxpoint, classifireID):
        # Detected Pice
        radius = self.getPointDistance(midpoint, maxpoint)
        cv2.circle(img, midpoint, int(radius), (0, 255, 0))

        # Orientation Pice
        radius = self.getPointDistance(midpoint, self.normedMaxPosition(midpoint, classifireID))
        cv2.circle(img, self.normedMaxPosition(midpoint, classifireID), 7, (0, 0, 255), -1)
        cv2.circle(img, midpoint, int(radius), (255, 0, 0))
        pass

    def editForTensorflow(self, img):
        img_as_string = cv2.imencode('.jpg', img)[1].tostring()
        return img_as_string

    # TODO x,y form Corner
    def getCorners(self, img):
        pass

    def getAllPices(self,path=None):
        # img_filtered, img_input = CameraManager.CameraManager().getImageFile(path)
        img_filtered, img_input = CameraManager.CameraManager().getCameraFrameInput()
        return self.extractPices(img_filtered, img_input)
