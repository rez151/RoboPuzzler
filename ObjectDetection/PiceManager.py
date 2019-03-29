import ObjectDetection.CameraManager as CameraManager
import cv2

class PiceManager():
    def extractPices(self, img_filtered):
        extractedPices = []
        _, cnts, _ = cv2.findContours(img_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            # Extract pice
            extractPice = img_filtered[y:y + h, x:x + w]
            # get Midpoint
            M = cv2.moments(ctr)
            if M["m10"] > 0:
                mX = int(M["m10"] / M["m00"])
                mY = int(M["m01"] / M["m00"])
            extractedPices.insert(i, [i, extractPice, mX, mY])
        return extractedPices

    def getAllPices(self):
        return  PiceManager().extractPices(CameraManager.CameraManager().getImageFile())



if __name__ == '__main__':
    cv2.imshow("original",CameraManager.CameraManager().getImageFile())
    for imageID, piceImg, mX, mY in PiceManager().getPices():
        cv2.imshow(str(imageID), piceImg)
        print("ID: " + str(imageID) + " X: " + str(mX) + " Y: " + str(mY))
    cv2.waitKey(0)
    cv2.destroyAllWindows()