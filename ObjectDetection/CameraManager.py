import cv2
import numpy as np
import ObjectDetection.MarkerTrackingManager as tm

class CameraManager:
    thresh_filter = 245

    def getImageFile(self, path):
        img_input = cv2.imread(path)
        thresh, gray = self.imageFilter(img_input)
        return thresh, img_input, gray

    def getImageFilebyID(self, id):
        img_input = cv2.imread("Rotation/"+id+".jpg")
        thresh = self.imageFilter(img_input)[0]
        return thresh, img_input

    def getCameraFrameInput(self, cameraindex):
        cap = cv2.VideoCapture(cameraindex)
        _, img_input = cap.read()
        img_input=cv2.resize(img_input,(1080,720))
        img_input = self.arucoMarkerCut(img_input, cameraindex)
        thresh, gray = self.imageFilter(img_input)
        # cv2.imshow("give me the fucking trash", thresh)
        return thresh, img_input, gray

    def arucoMarkerCut(self, img, cameraindex):
        try:
            corners = tm.MarkerTrackingManager().getMarkerPoints(cameraindex)[0]
            if (len(corners) == 4):
                image_width = int(1080)
                image_hight = int(720)
                pts1 = np.float32(corners)
                pts2 = np.float32([[0, 0], [image_width, 0], [0, image_hight], [image_width, image_hight]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                img_input = cv2.warpPerspective(img, M, (image_width, image_hight))
        except Exception as e:
            print(e)
        return img

    def imageFilter(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray, (23, 23), 0)
        thresh = cv2.threshold(gray2, self.thresh_filter, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(thresh, None, iterations=0)
        thresh = cv2.dilate(thresh, None, iterations=0)
        return thresh, gray

    def setThresh_filter(self,value):
        self.thresh_filter = value


if __name__ == '__main__':
    thresh, image, gray = CameraManager().getCameraFrameInput(1)
    cv2.imshow("Thresh", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()