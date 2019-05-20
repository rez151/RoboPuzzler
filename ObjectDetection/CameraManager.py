import cv2
import numpy as np
import ObjectDetection.trackMarker as tm

class CameraManager:
    thresh_filter = 240


    def getImageFile(self, path):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray2, self.thresh_filter, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=0)
        return thresh, image, gray

    def getImageFilebyID(self, id):
        image = cv2.imread("Rotation/"+id+".jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        return thresh, image

    def getCameraFrameInput(self):
        cap = cv2.VideoCapture(1)
        _, img_input = cap.read()

        if(tm.trackMarker().getMarker().__sizeof__()>3):
            image_width = int(2070 /2)
            image_hight = int(1680 /2)
            pts1 = np.float32((tm.trackMarker().getMarker()))
            # pts1 = np.sort(pts1,0)
            print(pts1)
            pts2 = np.float32([[0, 0], [image_width, 0], [0, image_hight], [image_width, image_hight]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img_input = cv2.warpPerspective(img_input, M, (image_width, image_hight))

        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray, (23, 23), 0)
        # gray = cv2.medianBlur(gray, 17)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 4)
        #thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_TRUNC)[1]
        thresh = cv2.threshold(gray2, self.thresh_filter, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(thresh, None, iterations=9)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cv2.imshow("give me the fucking trash", thresh)
        return thresh, img_input, gray


    def setThresh_filter(self,value):
        self.thresh_filter = value

    def getExtractPice(self, img_filtered, ctr):
        x, y, w, h = cv2.boundingRect(ctr)
        extractPice = img_filtered[y:y + h, x:x + w]
        return extractPice





if __name__ == '__main__':
    thresh, image, gray = CameraManager().getImageFile('TestImages/testwithallpicesnorotated.jpg')

    cv2.imshow("Thresh", thresh)
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