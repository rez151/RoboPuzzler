import cv2
import numpy as np

def GetImage():
    image = cv2.imread("Bilder/moreshapes.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    median = cv2.medianBlur(thresh, 5)
    median = cv2.bitwise_not(median);
    edges = cv2.Canny(median, 100, 100, apertureSize=3)
    return image,median;

def GetContours(image, median):
    pice = []
    cnts, _ = cv2.findContours(median, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi = median[y:y + h, x:x + w]
        coords = np.column_stack(np.where(roi > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        #cv2.imwrite('extraktPices/{}.png'.format(i), roi)

        cv2.drawContours(image, [ctr], 0, (0, 0, 255), 1)
        M = cv2.moments(ctr)
        if M["m10"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)

        (h, w) = roi.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(roi, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        pice.insert(i, [i,roi,cX,cY,angle,rotated])

    return pice,image;


def main():
    image, median = GetImage()
    pice, imageContours = GetContours(image,median)
    cv2.imshow("shapes", imageContours)

    for imageID, piceImg, mX, mY, r, rImg in pice:
        cv2.imshow(str(imageID), piceImg)
        print("ID: "+str(imageID) + " X: "+str(mX)+" Y: "+str(mY)+" R: "+str(r))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__== "__main__":
  main()
