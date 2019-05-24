import cv2
import numpy as np
import ObjectDetection.MarkerTrackingManager as tm

class CameraManager:
    thresh_filter = 245


    def getImageFile(self, path):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray, (17, 17), 0)
        thresh = cv2.threshold(gray2, self.thresh_filter, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=0)
        thresh = cv2.dilate(thresh, None, iterations=7)
        return thresh, image, gray

    def getImageFilebyID(self, id):
        image = cv2.imread("Rotation/"+id+".jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=0)
        thresh = cv2.dilate(thresh, None, iterations=0)
        return thresh, image

    def getCameraFrameInput(self,cameraIndex):
        cap = cv2.VideoCapture(cameraIndex)
        _, img_input = cap.read()
        img_input=cv2.resize(img_input,(1080,720))
        try:
            corners = tm.MarkerTrackingManager().getMarkerPoints(1)[0]
            if(len(corners)==4):
                image_width = int(1080)
                image_hight = int(720)
                pts1 = np.float32(corners)
                pts2 = np.float32([[0, 0], [image_width, 0], [0, image_hight], [image_width, image_hight]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                img_input = cv2.warpPerspective(img_input, M, (image_width, image_hight))
        except Exception as e:
            print(e)

        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray, (23, 23), 0)
        # gray = cv2.medianBlur(gray, 17)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 4)
        #thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_TRUNC)[1]
        thresh = cv2.threshold(gray2, self.thresh_filter, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(thresh, None, iterations=0)
        thresh = cv2.dilate(thresh, None, iterations=0)
        cv2.imshow("give me the fucking trash", thresh)
        return thresh, img_input, gray


    def setThresh_filter(self,value):
        self.thresh_filter = value

    def getExtractPice(self, img_filtered, ctr):
        x, y, w, h = cv2.boundingRect(ctr)
        extractPice = img_filtered[y:y + h, x:x + w]
        return extractPice





if __name__ == '__main__':
    thresh, image, gray = CameraManager().getCameraFrameInput(1)

    cv2.imshow("Thresh", image)
    # thresh, image, gray = CameraManager().getCameraFrameInput()
    # imageshow = image.copy()
    #
    # _, cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
    #
    # for i, ctr in enumerate(sorted_ctrs):
    #     if (i == 0):
    #         pass
    #     else:
    #         cv2.drawContours(imageshow, [ctr], 0, (0, 0, 255), 2)
    #         extractPice = CameraManager().getExtractPice(image, ctr)
    #         cv2.imwrite("Images/test/" + str(i) + ".jpg", extractPice)
    # cv2.imshow("Demo", imageshow)
    #
    cv2.waitKey(0)
    cv2.destroyAllWindows()