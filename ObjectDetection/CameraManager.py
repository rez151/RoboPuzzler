import cv2
import numpy as np
import ObjectDetection.MarkerTrackingManager as tm

class CameraManager:
    thresh_filter = 245

    def getImagebyFile(self, path):
        img_input = cv2.imread(path)
        thresh, gray = self.imageFilter(img_input)
        return thresh, img_input, gray

    def getImageFilebyID(self, id):
        img_input = cv2.imread("Rotation/"+id+".jpg")
        thresh = self.imageFilter(img_input)[0]
        return thresh

    def getCameraFrameInput(self, cameraindex):
        cap = cv2.VideoCapture(cameraindex)
        print("bild gemacht")
        _, img_input = cap.read()
        img_input = self.arucoMarkerCut(img_input, cameraindex)
        thresh, gray = self.imageFilter(img_input)
        #optional output
        cv2.imshow("give me the fucking trash", thresh)
        return thresh, img_input, gray

    def arucoMarkerCut(self, img, cameraindex):
        try:
            img = cv2.resize(img, (1920, 1080))
            corners = tm.MarkerTrackingManager().getMarkerPoints(cameraindex)[0]
            if (len(corners) == 4):
                image_width = int(1920)
                image_hight = int(1080)
                pts1 = np.float32(corners)
                pts2 = np.float32([[0, 0], [image_width, 0], [0, image_hight], [image_width, image_hight]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                img = cv2.warpPerspective(img, M, (image_width, image_hight))
        except Exception as e:
            print(e)
        return img

    def imageFilter(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (23, 23), 0)
        thresh = cv2.threshold(blur, self.thresh_filter, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(thresh, None, iterations=0)
        thresh = cv2.dilate(thresh, None, iterations=0)
        return thresh, gray

    def setThresh_filter(self,value):
        self.thresh_filter = value

    def getMidpoint(self, ctr):
        M = cv2.moments(ctr)
        if M["m10"] > 0:
            mX = int(M["m10"] / M["m00"])
            mY = int(M["m01"] / M["m00"])
            return mX, mY


if __name__ == '__main__':
    thresh, image, gray = CameraManager().getCameraFrameInput(0)
    # thresh= cv2.resize(thresh, (2268, 1535))
    # image= cv2.resize(image, (2268, 1535))
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]

    try:
        for i, ctr in enumerate(cnts):
            if (i == (len(cnts) - 1)):
                print(len(cnts))
                print("Done  \n")
            else:
                # get Extracted pice
                midpoint = CameraManager().getMidpoint(ctr)
                cv2.circle(image, midpoint, 7, (0, 255, 0), -1)
                cv2.drawContours(image, [ctr], 0, (0, 0, 255), 2)
                print(str(midpoint[0]/60) + ", " + str(midpoint[1]/60))
    except Exception as e:
        print(e)

    cv2.imshow("Thresh", cv2.resize(image, (1080, 720)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
