import cv2
import numpy as np
import ObjectDetection.MarkerTrackingManager as MarkerTrackingManager
import ObjectDetection.PiceManager as PiceManager
import ObjectDetection.MathManager as MathManager


class CameraManager:
    thresh_filter = 245

    def getImagebyFile(self, path):
        img_input = cv2.imread(path)
        thresh, gray = self.imageFilter(img_input)
        return thresh, img_input, gray

    def getImageFilebyID(self, id):
        img_input = cv2.imread("Rotation/" + str(id) + ".jpg")
        thresh = self.imageFilter(img_input)[0]
        return thresh

    @staticmethod
    def getCameraFrameInput(cameraindex):
        cap = cv2.VideoCapture(cameraindex)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret, img_input = cap.read()
        return img_input

    @staticmethod
    def getUndistortImg(img):
        calib_path = "CameraCalibration"
        camera_matrix = np.loadtxt(calib_path + '/cameraMatrix.txt', delimiter=',')
        camera_distortion = np.loadtxt(calib_path + '/cameraDistortion.txt', delimiter=',')
        img_undistorted = cv2.undistort(img, camera_matrix, camera_distortion)
        return img_undistorted

    def getAreaOfInterest(self, cameraindex):
        img_input = self.getCameraFrameInput(cameraindex)
        img_input = self.getUndistortImg(img_input)
        img_input = self.arucoMarkerCut(img_input)
        thresh, gray = self.imageFilter(img_input)
        return thresh, img_input, gray

    @staticmethod
    def arucoMarkerCut(img):
        try:
            img = cv2.resize(img, (1920, 1080))
            corners, areasize, _ = MarkerTrackingManager.MarkerTrackingManager().getMarkerPoints(img)
            if len(corners) == 4:
                image_width = int(areasize[0])
                image_hight = int(areasize[1])
                pts1 = np.float32(corners)
                pts2 = np.float32([[0, 0], [image_width, 0], [0, image_hight], [image_width, image_hight]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                img = cv2.warpPerspective(img, M, (image_width, image_hight))
        except Exception as e:
            print("Error: arucoMarkerCut, " + str(e))
        return img

    def imageFilter(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (23, 23), 0)
        thresh = cv2.threshold(blur, self.thresh_filter, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(thresh, None, iterations=0)
        thresh = cv2.dilate(thresh, None, iterations=0)
        return thresh, gray

    def setThresh_filter(self, value):
        self.thresh_filter = value

    def saveExtractImages(self):
        thresh, image, gray = self.getAreaOfInterest(1)
        cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        try:
            for i, ctr in enumerate(cnts):
                if i == (len(cnts) - 1):
                    print("Done  \n")
                else:
                    extractPice = PiceManager.PiceManager().getExtractPice(image, ctr)
                    path = "Images/test/" + str(i) + ".jpg"
                    cv2.imwrite(path, cv2.resize(extractPice, (224, 224)))
                    print("Saved " + str(i) + "picture to "+path)
            print("Saved all extract pictures")
        except Exception as e:
            print("Error: " + e)


if __name__ == '__main__':
    thresh, image, gray = CameraManager().getAreaOfInterest(1)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

    try:
        for i, ctr in enumerate(cnts):
            if i == (len(cnts) - 1):
                print("Done  \n")
            else:
                extractPice = PiceManager.PiceManager().getExtractPice(thresh, ctr)
                cv2.imwrite("Images/" + str(i) + ".jpg", extractPice)
                midpoint = MathManager.MathManager.getPiceMidpoint(ctr)
                midpointCm = MathManager.MathManager().getPointToCm(ctr)
                print(str(midpointCm[0]) + ", " + str(midpointCm[1]))
    except Exception as e:
        print(e)

    cv2.imshow("Thresh", cv2.resize(image, (1015, 734)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
