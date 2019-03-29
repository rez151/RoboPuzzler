import cv2


class CameraManager:

    def getImageFile(self):
        img_input = cv2.imread("Bilder/moreshapes.jpg")
        img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        _, img_filtered = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY)
        img_filtered = cv2.medianBlur(img_filtered, 5)
        img_filtered = cv2.bitwise_not(img_filtered);
        return img_filtered;

    def getCameraInput(self):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            _, img_input = cap.read()
            cap.release()
            if img_input is not None:
                img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
                _, img_filtered = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY)
                img_filtered = cv2.medianBlur(img_filtered, 5)
                img_filtered = cv2.bitwise_not(img_filtered)
                return img_filtered