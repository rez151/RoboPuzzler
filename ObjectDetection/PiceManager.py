import ObjectDetection.CameraManager as CameraManager
import ObjectDetection.Classifier as Classifier
import cv2
import math
import numpy as np

class PiceManager:
    def extractPices(self, img_filtered, img_input):
        extractedPices = []
        _, cnts, _ = cv2.findContours(img_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for i, ctr in enumerate(sorted_ctrs):
            if(i==0):
                pass
            else:
                # get Extracted pice
                extractPice = PiceManager().getExtractPice(img_filtered,ctr)
                midpoint = PiceManager().getMidpoint(ctr)
                maxpoint = PiceManager().getPointMaxDistance(midpoint,ctr)
                classifierID = Classifier.Classifier().Classifier(PiceManager().editForTensorflow(extractPice))
                normedmaxpoint = PiceManager().normedMaxPosition(midpoint,classifierID)
                rotation = PiceManager().getRotation(midpoint, maxpoint, normedmaxpoint)

                # draw Contours
                cv2.drawContours(img_input, [ctr], 0, (0, 0, 255), 2)
                # draw Midpoint
                cv2.circle(img_input, midpoint, 7, (0, 255, 0), -1)
                # draw MaxPoint
                cv2.circle(img_input, maxpoint, 7, (0, 255, 0), -1)
                # draw Classification
                cv2.putText(img_input, str(classifierID), midpoint, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                #draw rotation Circles
                PiceManager().drawRotationCircle(img_input,midpoint,maxpoint,classifierID)
                print(str(int((i*100)/(sorted_ctrs.__len__()-1))) +"% Done")
                extractedPices.insert(i, [i, extractPice, midpoint, str(classifierID), rotation])
        return extractedPices, img_input

    def getExtractPice(self, img_filtered, ctr):
        x, y, w, h = cv2.boundingRect(ctr)
        extractPice = img_filtered[y:y + h, x:x + w]
        return extractPice

    def getExtractPiceByMidpoint(self,img_filtered,midpoint,pointWithMaxDistance):
        x, y = midpoint
        w, h = pointWithMaxDistance
        extractPice = img_filtered[y:(y) + h, x:(x) + w]
        cv2.imshow("w",extractPice)
        return extractPice

    def getContour(self,img):
        _, cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for i, ctr in enumerate(sorted_ctrs):
            if (i == 0):
                pass
            else:
                return ctr


    def getOrientationPoint(self,id):
        img_filtered, img_input = CameraManager.CameraManager().getImageFilebyPath(id)
        normedPiceContour  = PiceManager().getContour(img_filtered)
        normedPice = PiceManager().getExtractPice(img_filtered,normedPiceContour)
        mx, my = PiceManager().getMidpoint(normedPiceContour)
        x, y = PiceManager().getPointMaxDistance((mx,my),normedPiceContour)
        cv2.imshow("o",normedPice)
        return x, y, mx, my

    def normedMaxPosition(self, midpoint, classifireID):
        ox, oy, omx, omy = PiceManager().getOrientationPoint(classifireID)
        mx, my = midpoint
        ox = (mx - omx) + ox
        oy = (my - omy) + oy
        return ox,oy


    def getRotation(self,midpoint,maxpoint,normedmaxpoint):
        r=1
        rotation = 0
        distance = PiceManager().getPointDistance(maxpoint,normedmaxpoint)
        while(r!=361):
            rpoint = PiceManager().rotatePoint(maxpoint,midpoint,r)
            rdistance = PiceManager().getPointDistance(rpoint,normedmaxpoint)
            if(distance>rdistance):
                distance= rdistance
                if(r>180):
                    rotation = 360-r
                else:
                    rotation = r
            r += 1
        return ((rotation/360)*100)#*(-1) korrekt rotation

    def getMidpoint(self, ctr):
        M = cv2.moments(ctr)
        if M["m10"] > 0:
            mX = int(M["m10"] / M["m00"])
            mY = int(M["m01"] / M["m00"])
            return  mX, mY

    def getPointDistance(self,p1,p2):
        return math.sqrt(math.pow((p2[1]-p1[1]),2)+math.pow((p2[0]-p1[0]),2))

    def getPointMaxDistance(self,midpoint,p):
        p = np.unique(p, axis=0)
        maxdist = 0
        maxpoint = {0,0}
        for p1 in p:
            for p2 in p1:
                if(maxdist<PiceManager().getPointDistance(midpoint,p2)):
                    maxdist=PiceManager().getPointDistance(midpoint,p2)
                    maxpoint = p2
        return maxpoint[0],maxpoint[1]

    def getPointMinDistance(self,midpoint,p):
        p = np.unique(p, axis=0)
        mindist = 0
        minpoint = {0,0}
        for p1 in p:
            for p2 in p1:
                if(mindist>PiceManager().getPointDistance(midpoint,p2)):
                    mindist=PiceManager().getPointDistance(midpoint,p2)
                    minpoint = p2
        return minpoint[0],minpoint[1]

    def rotatePoint(self,p,midpoint,a):
        x0, y0 = p
        x1, y1 = midpoint
        x2 = ((x1 - x0) * math.cos(a)) - ((y1 - y0) * math.sin(a)) + x0;
        y2 = ((x1 - x0) * math.sin(a)) + ((y1 - y0) * math.cos(a)) + y0;
        return x2, y2

    def drawRotationCircle(self,img,midpoint,maxpoint,classifireID):
        #Detected Pice
        radius = PiceManager().getPointDistance(midpoint,maxpoint)
        cv2.circle(img,midpoint,int(radius),(0, 255, 0))

        #Orientation Pice
        radius = PiceManager().getPointDistance(midpoint, PiceManager().normedMaxPosition(midpoint, classifireID))
        cv2.circle(img, PiceManager().normedMaxPosition(midpoint, classifireID), 7, (0, 0, 255), -1)
        cv2.circle(img, midpoint, int(radius), (255, 0, 0))
        pass



    def editForTensorflow(self,img):
        img_as_string = cv2.imencode('.jpg', img)[1].tostring()
        return img_as_string


    # TODO x,y form Corner
    def getCorners(self, img):
        pass


    def getAllPices(self):
        img_filtered, img_input = CameraManager.CameraManager().getImageFile()
        # img_filtered, img_input = CameraManager.CameraManager().getCameraFrameInput()
        return PiceManager().extractPices(img_filtered, img_input)


