import cv2

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
        if cap.isOpened():
            _, img_input = cap.read()
            cap.release()
            if img_input is not None:
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




if __name__ == '__main__':
    cv2.imshow("Demo",cv2.resize(CameraManager().getCameraFrameInput()[0], (1080, 720)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
