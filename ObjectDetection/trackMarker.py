
import cv2
from ObjectDetection.lib import tracker
import math
import numpy as np
from numpy  import array


class trackMarker:

    BLUE = (255, 50, 50)
    GREEN = (50, 255, 50)
    RED = (50, 50, 255)
    WHITE = (255, 255, 255)


    def getMarker(self):
        cap = cv2.VideoCapture(1)
        #cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        __, img = cap.read()

        markers = tracker.find_markers(img)
        midpointlist = list()

        for m_id, marker in markers.items():
            midpoint = self.getMidpoint(marker.contour)
            cv2.drawContours(img, [marker.contour], -1, self.GREEN, 2)
            cv2.line(img, marker.position, marker.major_axis, self.WHITE, 2)
            cv2.line(img, marker.position, marker.minor_axis, self.WHITE, 2)
            cv2.circle(img, midpoint,2,self.RED,-1)
            midpointlist.append(list(midpoint))
        cv2.imshow('Main window', cv2.resize(img.copy(), (800,600)))
        return self.fourPoints(midpointlist)

    def fourPoints(self,midpointlist):
        distanceList = list()
        try:
            for p1 in midpointlist:
                for p2 in midpointlist:
                    if(p1!=p2):
                        distance = self.getPointDistance(p1,p2)
                        distanceList.append([int(distance),p1,p2])

            distanceList.sort()

            pointList = list()

            pointList.append(distanceList[len(distanceList)-1][1])
            pointList.append(distanceList[len(distanceList)-1][2])
            pointList.append(distanceList[len(distanceList)-3][1])
            pointList.append(distanceList[len(distanceList)-3][2])

            outlist = list()
            # add wight
            for point in pointList:
                out = [int(self.getPointDistance([0,0],point)), [point]]
                outlist.append(out)

            outlist.sort()

            p1 = outlist.pop(0)[1]
            p4 = outlist.pop(len(outlist) - 1)[1]
            if (outlist[0][1][0] < outlist[1][1][0]):
                p2 = outlist.pop(1)[1]
                p3 = outlist.pop(0)[1]
            else:
                p3 = outlist.pop(1)[1]
                p2 = outlist.pop(0)[1]

            return [p1, p2, p3, p4]
        except Exception:
            pass
    pass





    def getPointDistance(self, p1, p2):
        return math.sqrt(math.pow((p2[1] - p1[1]), 2) + math.pow((p2[0] - p1[0]), 2))


    def getMidpoint(self,ctr):
        M = cv2.moments(ctr)
        if M["m10"] > 0:
            mX = int(M["m10"] / M["m00"])
            mY = int(M["m01"] / M["m00"])
            return mX, mY

if __name__ == '__main__':
    while True:

        trackMarker().getMarker()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cv2.destroyAllWindows()