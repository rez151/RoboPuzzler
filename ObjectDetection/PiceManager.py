import cv2
import imutils
from ObjectDetection.CameraManager import CameraManager
from ObjectDetection.Classifier import Classifire
from ObjectDetection.MathManager import MathManager


class PiceManager:
    def extractPices(self, img_thresh, img_input, gray):
        extractedPices = []
        cnts = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]

        image = img_input.copy()
        for i, ctr in enumerate(cnts):
            if i == len(cnts) - 1:
                print(len(cnts))
                print("Done  \n")
            else:
                # get Extracted pice
                extractPiceClassification = self.getExtractPice(gray, ctr)
                midpoint = MathManager.getMidpoint(ctr)
                maxpoint = MathManager().getPointMaxDistance(midpoint, ctr)
                classifierID, id = Classifire().Classifier(extractPiceClassification)
                normedmaxpoint = MathManager().normedMaxPosition(midpoint, classifierID)
                rotation = self.getRotation(id)
                # correct Rotation
                extractPiceClassification = imutils.rotate_bound(extractPiceClassification, rotation)
                extractedPices.insert(i, [extractPiceClassification, midpoint, maxpoint, str(id), classifierID, normedmaxpoint, rotation, ctr])
                # print progress status
                print(str(int((i * 100) / (len(cnts) - 1))) + "%")
        image = self.drawInformations(image)
        return extractedPices, image

    @staticmethod
    def getExtractPice(img_filtered, ctr):
        x, y, w, h = cv2.boundingRect(ctr)
        extractPice = img_filtered[y:y + h, x:x + w]
        return extractPice

    @staticmethod
    def getContour(img):
        cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        sorted_ctrs = sorted(cnts, key=lambda ct: cv2.boundingRect(ctr)[0])

        for i, ctr in enumerate(sorted_ctrs):
            if i == 0:
                pass
            else:
                return ctr

    @staticmethod
    def getRotation(id):
        img_thresh = CameraManager.getImageFilebyID(id)
        ctr = PiceManager.getContour(img_thresh)
        return MathManager().getOrientationPoint(ctr)

    @staticmethod
    def getCorners(img):
        return cv2.cornerHarris(img, 2, 3, 0.04)

    @staticmethod
    def drawRotationCircle(img, midpoint, maxpoint, normedmaxpoint):
        # Detected Pice
        radius = MathManager().getRadius(midpoint, maxpoint)
        cv2.circle(img, midpoint, int(radius), (0, 255, 0))
        # Orientation Pice
        radius = MathManager.getPointDistance(midpoint, normedmaxpoint)
        cv2.circle(img, normedmaxpoint, 7, (0, 0, 255), -1)
        cv2.circle(img, midpoint, int(radius), (255, 0, 0))
        pass

    @staticmethod
    def drawInformations(image, extractedPices):
        for pices in extractedPices:
            extractPiceClassification, midpoint, maxpoint, id, classifierID, normedmaxpoint, rotation, ctr = pices
            # draw line from normedpoint and local maxpoint to midpoint
            cv2.line(image, midpoint, normedmaxpoint, (255, 0, 0), 1)
            cv2.line(image, midpoint, maxpoint, (0, 255, 0), 1)
            # draw Contours
            cv2.drawContours(image, [ctr], 0, (0, 0, 255), 2)
            # draw Midpoint
            cv2.circle(image, midpoint, 7, (0, 255, 0), -1)
            # draw MaxPoint
            cv2.circle(image, maxpoint, 7, (0, 255, 0), -1)
            # draw Classification text
            cv2.putText(image, (str(classifierID)), midpoint, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
            # draw rotation Circles
            PiceManager.drawRotationCircle(image, midpoint, maxpoint, normedmaxpoint)
        return image

    def getAllPicesyPath(self, path=None):
        img_filtered, img_input, gray = CameraManager.getImagebyFile(path)
        return self.extractPices(img_filtered, img_input, gray)

    def getAllPicesbyFrame(self, cameraIndex):
        img_filtered, img_input, gray = CameraManager.getCameraFrameInput(cameraIndex)
        return self.extractPices(img_filtered, img_input, gray)
