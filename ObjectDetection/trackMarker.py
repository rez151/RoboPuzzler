
import cv2
from ObjectDetection.lib import tracker


class trackMarker:

    BLUE = (255, 50, 50)
    GREEN = (50, 255, 50)
    RED = (50, 50, 255)
    WHITE = (255, 255, 255)


    def getMarker(self):
        cap = cv2.VideoCapture(1)
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
        print(midpointlist)
        cv2.imshow('Main window', cv2.resize(img.copy(), (600,400)))
        return midpointlist



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