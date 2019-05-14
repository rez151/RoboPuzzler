import cv2
import numpy as np
import ObjectDetection.trackMarker as tm

class CameraManager:
    thresh_filter = 180


    def getImageFile(self, path):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, self.thresh_filter, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cv2.imshow("Thresh", thresh)

        return thresh, image

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
        try:
            if(tm.trackMarker().getMarker().__sizeof__()>3):
                image_width = 400
                image_hight = 300
                pts1 = np.float32((tm.trackMarker().getMarker()))
                print(pts1)
                pts2 = np.float32([[0, 0], [image_width, 0], [0, image_hight], [image_width, image_hight]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                img_input = cv2.warpPerspective(img_input, M, (image_width, image_hight))
        except: Exception

        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (17, 17), 0)
        # gray = cv2.medianBlur(gray, 17)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 4)
        #thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_TRUNC)[1]
        thresh = cv2.threshold(gray, self.thresh_filter, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.erode(thresh, None, iterations=8)
        thresh = cv2.dilate(thresh, None, iterations=2)
        return thresh, img_input


    def setThresh_filter(self,value):
        self.thresh_filter = value

    def getExtractPice(self, img_filtered, ctr):
        x, y, w, h = cv2.boundingRect(ctr)
        extractPice = img_filtered[y:y + h, x:x + w]
        return extractPice



if __name__ == '__main__':

    thresh,image = CameraManager().getCameraFrameInput()


    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        if (i == 0):
            pass
        else:
            cv2.drawContours(image, [ctr], 0, (0, 0, 255), 2)
            extractPice = CameraManager().getExtractPice(thresh,ctr)
            # cv2.imwrite("Images/test/"+str(i)+".jpg",extractPice)
    cv2.imshow("Demo", image)
    cv2.imshow("Demo2", thresh)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

