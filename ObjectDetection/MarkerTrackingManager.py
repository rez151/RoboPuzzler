import numpy as np
import cv2
import cv2.aruco as aruco
from ObjectDetection.MathManager import MathManager


class MarkerTrackingManager:
    def getMarkerPoints(self, gray):
        gray = cv2.cvtColor(gray.copy(), cv2.COLOR_BGR2GRAY)
        returnPoints = []

        # --- Get the camera calibration path
        calib_path = "CameraCalibration"
        camera_matrix = np.loadtxt(calib_path + '/cameraMatrix.txt', delimiter=',')
        camera_distortion = np.loadtxt(calib_path + '/cameraDistortion.txt', delimiter=',')

        # --- 180 deg rotation matrix around the x axis
        R_flip = np.zeros((3, 3), dtype=np.float32)
        R_flip[0, 0] = 1.0
        R_flip[1, 1] = -1.0
        R_flip[2, 2] = -1.0

        # --- Define the aruco dictionary
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        parameters = aruco.DetectorParameters_create()

        # -- Find all the aruco markers in the image
        corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                                                     cameraMatrix=camera_matrix, distCoeff=camera_distortion)

        for p in corners:
            for p1 in p:
                returnPoints.append([int(p1[0][0]), int(p1[0][1])])

        returnPoints = self.getfourPoints(returnPoints)

        self.drawArucoMarker(ids, corners, camera_matrix, camera_distortion, gray)

        areasize = self.getAreaSize(returnPoints)
        return returnPoints, areasize, gray

    def getfourPoints(self, midpointlist):
        pointList = self.findOuterRectanglePoints(midpointlist)

        outlist = list()
        # add wight dependending on distance to (0, 0)
        for point in pointList:
            out = [int(MathManager.getPointDistance([0, 0], point)), [point]]
            outlist.append(out)

        outlist.sort()

        p1 = outlist.pop(0)[1]
        p4 = outlist.pop(len(outlist) - 1)[1]
        if outlist[0][1][0] < outlist[1][1][0]:
            p2 = outlist.pop(1)[1]
            p3 = outlist.pop(0)[1]
        else:
            p3 = outlist.pop(1)[1]
            p2 = outlist.pop(0)[1]

        return [p1[0], p2[0], p3[0], p4[0]]

    @staticmethod
    def findOuterRectanglePoints(poinlist):
        distanceList = list()
        pointList = list()
        for p1 in poinlist:
            for p2 in poinlist:
                if p1 != p2:
                    distance = MathManager.getPointDistance(p1, p2)
                    distanceList.append([int(distance), p1, p2])

        distanceList.sort()

        pointList.append(distanceList[len(distanceList) - 1][1])
        pointList.append(distanceList[len(distanceList) - 1][2])
        pointList.append(distanceList[len(distanceList) - 3][1])
        pointList.append(distanceList[len(distanceList) - 3][2])

        return pointList

    @staticmethod
    def drawArucoMarker(ids, corners, camera_matrix, camera_distortion, img_input):
        # --- Define Tag
        id_to_find = [72]
        marker_size = 1.5  # [cm]
        if ids is not None:
            for id in ids:
                if id_to_find.count(id) == 1:
                    ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

                    # -- Unpack the output, get only the first
                    rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

                    # -- Draw the detected marker and put a reference frame over it
                    aruco.drawDetectedMarkers(img_input, corners, ids)
                    aruco.drawAxis(img_input, camera_matrix, camera_distortion, rvec, tvec, 10)
        pass

    @staticmethod
    def getAreaSize(areapoints):
        p1 = areapoints[0]
        p2 = areapoints[1]
        p3 = areapoints[2]
        x = MathManager.getPointDistance(p1, p3)
        y = MathManager.getPointDistance(p1, p2)
        return x, y


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, img_input = cap.read()
    returnPoints, areasize, img_input = MarkerTrackingManager().getMarkerPoints(img_input)
    print(returnPoints)
    cv2.imshow('frame', cv2.resize(img_input, (1080, 720)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
