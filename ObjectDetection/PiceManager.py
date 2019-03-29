import ObjectDetection.CameraManager as CameraManager
import ObjectDetection.Classifier as Classifier
import cv2

class PiceManager():
    def extractPices(self, img_filtered, img_input):
        extractedPices = []
        _, cnts, _ = cv2.findContours(img_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            # Extract pice
            extractPice = img_filtered[y:y + h, x:x + w]
            #draw Contours
            cv2.drawContours(img_input, [ctr], 0, (0, 0, 255), 1)
            # get Midpoint
            M = cv2.moments(ctr)
            if M["m10"] > 0:
                mX = int(M["m10"] / M["m00"])
                mY = int(M["m01"] / M["m00"])
                cv2.circle(img_input, (mX, mY), 7, (0, 0, 255), -1)

            #Count corner
            approx = cv2.approxPolyDP(ctr, 0.01*cv2.arcLength(ctr, True), True)
            #Classifiere Image
            classifieredID = Classifier.Classifier().piceClassifier(len(approx))
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img_input, str(classifieredID), (mX, mY), font, 2, (0, 255, 255))


            extractedPices.insert(i, [i, extractPice, mX, mY, str(classifieredID)])

        return extractedPices, img_input

    def getAllPices(self):
        #img_filtered, img_input = CameraManager.CameraManager().getImageFile()
        img_filtered, img_input = CameraManager.CameraManager().getCameraInput()
        return  PiceManager().extractPices(img_filtered,img_input)



if __name__ == '__main__':
    extractedPices, img_input = PiceManager().getAllPices()
    cv2.imshow("Original", img_input)
    cv2.imshow("filter", CameraManager.CameraManager().getCameraInput()[0])

    for imageID, piceImg, mX, mY, classifieredID in extractedPices:
        #cv2.imshow(str(imageID)+classifieredID, piceImg)
        print("ID: " + str(imageID) + " X: " + str(mX) + " Y: " + str(mY)+ " C: "+classifieredID)
    cv2.waitKey(0)
    cv2.destroyAllWindows()