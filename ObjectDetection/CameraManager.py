import cv2

class CameraManager:

    def getImageFile(self):
        image = cv2.imread("Rotation/Krokodil.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        return thresh, image

    def getImageFilebyPath(self,path):
        image = cv2.imread("Rotation/"+path+".jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
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
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                # threshold the image, then perform a series of erosions +
                # dilations to remove any small regions of noise
                thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.erode(thresh, None, iterations=2)
                thresh = cv2.dilate(thresh, None, iterations=2)
                return thresh, img_input
