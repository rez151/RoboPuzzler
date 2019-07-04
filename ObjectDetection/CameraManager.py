import cv2
import numpy as np
import ObjectDetection.MarkerTrackingManager as MarkerTrackingManager
import ObjectDetection.PiceManager as PiceManager


class CameraManager:
    def getImageByFile(self, path):
        img_input = cv2.imread(path)
        thresh, gray = self.imageFilter(img_input)
        return thresh, img_input, gray

    def getImageByID(self, id):
        img_input = cv2.imread("Rotation/" + str(id) + ".jpg")
        thresh, gray = self.imageFilter(img_input)
        return  thresh, gray

    @staticmethod
    def getImageByCamera(cameraindex):
        cap = cv2.VideoCapture(cameraindex)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        img_input = cap.read()[1]
        cap.release()
        return img_input

    @staticmethod
    def getUndistortImg(img):
        calib_path = "CameraCalibration"
        camera_matrix = np.loadtxt(calib_path + '/cameraMatrix.txt', delimiter=',')
        camera_distortion = np.loadtxt(calib_path + '/cameraDistortion.txt', delimiter=',')
        img_undistorted = cv2.undistort(img, camera_matrix, camera_distortion)
        return img_undistorted

    def getAreaOfInterest(self, cameraindex):
        img_input = self.getImageByCamera(cameraindex)
        img_input = self.getUndistortImg(img_input)
        img_input = self.arucoMarkerCut(img_input)
        thresh, gray = self.imageFilter(img_input)
        return thresh, img_input, gray

    @staticmethod
    def arucoMarkerCut(img_input):
        try:
            corners, areasize, _ = MarkerTrackingManager.MarkerTrackingManager().getMarkerPoints(img_input)
            if len(corners) == 4:
                image_width = int(areasize[1])
                image_hight = int(areasize[0])
                pts1 = np.float32(corners)
                pts2 = np.float32([[0, 0], [image_width, 0], [0, image_hight], [image_width, image_hight]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                img_input = cv2.warpPerspective(img_input, M, (image_width, image_hight))
        except Exception as e:
            print("Error: arucoMarkerCut, " + str(e))
        return img_input

    @staticmethod
    def imageFilter(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (23, 23), 0)
        thresh = cv2.threshold(blur, 245, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh, gray

    def saveExtractImages(self):
        thresh, image, gray = self.getAreaOfInterest(1)
        cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        try:
            for i, ctr in enumerate(cnts):
                if i == (len(cnts) - 1):
                    print("Done  \n")
                else:
                    extractPice = PiceManager.PiceManager().getExtractPice(image, ctr)
                    path = "Rotation/" + str(i) + ".jpg"
                    cv2.imwrite(path, extractPice)  # cv2.resize(extractPice, (224, 224)
                    print("Saved " + str(i) + " picture to "+path)
            print("Saved all extract pictures")
        except Exception as e:
            print("Error: " + e)


if __name__ == '__main__':
    # img = CameraManager().getImageByCamera(1)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = CameraManager().getImageByCamera(1)
    cv2.imshow("img", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()