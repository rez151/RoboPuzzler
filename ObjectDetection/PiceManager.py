import cv2
import imutils
from ObjectDetection.CameraManager import CameraManager
from ObjectDetection.Classifier import Classifire
from ObjectDetection.MathManager import MathManager


class PiceManager:
    def extractPices(self, img_thresh, img_input, gray):
        extractedPices = []
        cnts = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        cv2.imshow("penis",img_thresh)

        image = img_input.copy()
        for i, ctr in enumerate(cnts):
            if cv2.contourArea(ctr) < 100:
                continue

            if i == len(cnts) - 1:
                print("Done  \n")
            else:
                # get Extracted pice
                extractPiceClassification = self.getExtractPice(gray, ctr)
                midpoint = MathManager.getPiceMidpoint(ctr)
                print(midpoint)
                midpointcm = MathManager.getPointToCm(midpoint)
                maxpoint = MathManager().getPointMaxDistance(midpoint, ctr)
                classifierID, id = Classifire().Classifier(extractPiceClassification)
                normedmaxpoint = self.getNormedMaxPoint(midpoint, classifierID)
                rotation = MathManager.getRotation(midpoint, maxpoint, normedmaxpoint)
                dimension = MathManager().getPiceMeasurement(ctr, image)
                print("{:.2f}mm".format(dimension[0]) + ", " + "{:.2f}mm".format(dimension[1]))

                # correct Rotation
                extractPiceClassification = imutils.rotate_bound(extractPiceClassification, rotation)
                extractedPices.insert(i, [extractPiceClassification, midpoint, midpointcm, maxpoint, str(id), classifierID, normedmaxpoint, rotation, ctr])
                # print progress status
                print(str(int((i * 100) / (len(cnts) - 1))) + "%")
        image = self.drawInformations(image, extractedPices)
        return extractedPices, image

    @staticmethod
    def getExtractPice(img_filtered, ctr):
        x, y, w, h = cv2.boundingRect(ctr)
        extractPice = img_filtered[y:y + h, x:x + w]
        return extractPice

    @staticmethod
    def getContour(img_thresh):
        cnts = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        for i, ctr in enumerate(cnts):
            if i == 0:
                pass
            else:
                return ctr


    def getNormedMaxPoint(self, midpoint, id):
        img_thresh = CameraManager().getImageFilebyID(id)
        ctr = self.getContour(img_thresh)
        return MathManager().normedMaxPosition(midpoint, ctr)

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
            _, midpoint, _, maxpoint, _, classifierID, normedmaxpoint, _, ctr = pices
            # draw line from normedpoint and local maxpoint to midpoint
            cv2.line(image, midpoint, normedmaxpoint, (255, 0, 0), 1)
            cv2.line(image, midpoint, maxpoint, (0, 255, 0), 1)
            # draw Contours
            cv2.drawContours(image, [ctr], 0, (0, 0, 255), 2)
            # draw Midpoint
            # cv2.circle(image, midpoint, 7, (255, 255, 0), -1)
            # draw MaxPoint
            cv2.circle(image, maxpoint, 7, (0, 255, 0), -1)
            # draw Classification text
            cv2.putText(image, (str(classifierID)), midpoint, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
            # draw rotation Circles
            PiceManager.drawRotationCircle(image, midpoint, maxpoint, normedmaxpoint)
        return image

    def getAllPicesbyPath(self, path=None):
        img_filtered, img_input, gray = CameraManager().getImagebyFile(path)
        return self.extractPices(img_filtered, img_input, gray)

    def getAllPicesbyFrame(self, cameraIndex):
        img_thresh, img_input, gray = CameraManager().getAreaOfInterest(cameraindex=cameraIndex)
        return self.extractPices(img_thresh, img_input, gray)

