import cv2
import ObjectDetection.PiceManager as pm


class Main:
    def startDetection(self):
        extractedPices, img_input = pm.PiceManager().getAllPices()
        cv2.imshow("Original", img_input)
        for imageID, piceImg, midPoint, classifierID, rotation in extractedPices:
            print("ID: " + str(imageID) + " X: " + str(midPoint[0]) + " Y: " + str(midPoint[1]) + " C: " + classifierID + " R: "+ str(rotation)+"°")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Main().startDetection()