import cv2
import imutils
from ObjectDetection.CameraManager import CameraManager
from ObjectDetection.Classifier import Classifire
from ObjectDetection.MathManager import MathManager


class PiceManager:
    def extractPices(self, img_thresh, img_input, gray):
        extractedPices = []
        cnts = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        image = img_input.copy()
        for i, ctr in enumerate(cnts):
            if cv2.contourArea(ctr) < 100:
                continue

            if i == len(cnts) - 1:
                print("Done  \n")
                continue
            else:
                # get Extracted pice
                extractPiceGray = self.getExtractPice(gray, ctr)
                extractPiceThresh = self.getExtractPice(img_thresh, ctr)
                extractedctr = self.getContour(extractPiceThresh)
                # corners = self.getCorners(extractPiceThresh, image)
                midpoint = MathManager.getPiceMidpoint(ctr)
                midpointmm = MathManager().getPointToMM(img_input, midpoint)
                classifierID, id = Classifire().Classifier(extractPiceGray)
                normpicethresh, normpicegray = CameraManager().getImageByID(classifierID)
                normedctr = self.getContour(normpicethresh)
                # normedcorners = self.getCorners(normpicethresh)
                normedmidpoint = MathManager.getPiceMidpoint(normedctr)

                heatmap = 0

                rotation = MathManager().getPiceRotation(ctr, id, image)
                dimension = MathManager().getPiceDimension(ctr, image)
                # normedctr = MathManager().getTransformedContour(midpoint, normedctr)

                # correct Rotation
                extractPiceGray = imutils.rotate_bound(extractPiceGray, rotation)
                extractedPices.insert(i, [extractPiceGray, midpoint, midpointmm, id, classifierID, rotation, ctr, heatmap])
                # print progress status
                print("{}%".format(str(int((i * 100) / (len(cnts) - 1)))))
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
            if i == len(cnts) -1:
                pass
            else:
                return ctr

    @staticmethod
    def getCorners(gray,img):
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst, None)

        print(dst)
        return dst

    @staticmethod
    def drawInformations(image, extractedPices):
        for pices in extractedPices:
            _, midpoint, _, _, classifierID, _, ctr, _ = pices
            # draw Contours
            cv2.drawContours(image, [ctr], 0, (0, 0, 255), 1)
            # draw Midpoint
            cv2.circle(image, midpoint, 4, (255, 255, 0), -1)
            # draw Classification text
            cv2.putText(image, (str(classifierID)), midpoint, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            # draw rotation Circles
            # PiceManager.drawRotationCircle(image, midpoint, maxpoint, normedmaxpoint)
        return image

    def getAllPicesbyPath(self, path):
        img_filtered, img_input, gray = CameraManager().getImageByFile(path)
        return self.extractPices(img_filtered, img_input, gray)

    def getAllPicesbyFrame(self, cameraIndex):
        img_thresh, img_input, gray = CameraManager().getAreaOfInterest(cameraindex=cameraIndex)
        return self.extractPices(img_thresh, img_input, gray)

