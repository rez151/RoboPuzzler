import ObjectDetection.CameraManager as CameraManager
import ObjectDetection.Classifier as Classifier
import cv2
import numpy as np


class PiceManager:
    def extractPices(self, img_filtered, img_input):
        extractedPices = []
        _, cnts, _ = cv2.findContours(img_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for i, ctr in enumerate(sorted_ctrs):
            # get Extracted pice
            extractPice = PiceManager().getExtractPice(img_filtered,ctr)
            # draw Contours
            cv2.drawContours(img_input, [ctr], 0, (0, 0, 255), 1)
            # draw Midpoint
            cv2.circle(img_input, (PiceManager().getMidpoint(ctr)), 7, (0, 255, 0), -1)
            # Count corner
            approx = cv2.approxPolyDP(ctr, 0.01 * cv2.arcLength(ctr, True), True)
            cv2.drawContours(img_input, approx, -1, (255, 0, 255), 7)
            # Classifier Image
            classifierID = Classifier.Classifier().Classifier(PiceManager().editForTensorflow(extractPice))
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img_input, str(classifierID), (PiceManager().getMidpoint(ctr)), font, 1, (0, 0, 255))

            extractedPices.insert(i, [i, extractPice, (PiceManager().getMidpoint(ctr)), str(classifierID)])

        return extractedPices, img_input

    def getExtractPice(self, img_filtered, ctr):
        x, y, w, h = cv2.boundingRect(ctr)
        extractPice = img_filtered[y:y + h, x:x + w]
        return extractPice

    def getMidpoint(self, ctr):
        M = cv2.moments(ctr)
        if M["m10"] > 0:
            mX = int(M["m10"] / M["m00"])
            mY = int(M["m01"] / M["m00"])
            return  mX, mY

    def editForTensorflow(self,img):
        # Format for the Mul:0 Tensor
        img2 = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        # Numpy array
        np_image_data = np.asarray(img2)
        # maybe insert float convertion here - see edit remark!
        np_final = np.expand_dims(np_image_data, axis=0)
        return  np_final


    # TODO x,y form Corner
    def getCorners(self, img):
        pass


    def getAllPices(self):
        img_filtered, img_input = CameraManager.CameraManager().getImageFile()
        # img_filtered, img_input = CameraManager.CameraManager().getCameraFrameInput()
        return PiceManager().extractPices(img_filtered, img_input)


if __name__ == '__main__':
    extractedPices, img_input = PiceManager().getAllPices()
    cv2.imshow("Original", img_input)
    # cv2.imshow("filter", CameraManager.CameraManager().getCameraFrameInput()[0])
    cv2.imshow("filter", CameraManager.CameraManager().getImageFile()[0])

    for imageID, piceImg, midPoint, classifierID in extractedPices:
        #cv2.imshow(str(imageID)+classifierID, piceImg)
        print("ID: " + str(imageID) + " X: " + str(midPoint[0]) + " Y: " + str(midPoint[1]) + " C: " + classifierID )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
