from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

class Classifire:

    def Classifier(self, img):
        image_width = 224
        image_hight = 224
        modelpath = "model/model.h5"
        image = self.resizeImageToInputSize(img, image_width, image_hight)

        #load model
        model = self.loadModel(modelpath)

        prediction = model.predict(image)

        id = prediction.argmax(1)[0]
        print("prediction in %: "+str(prediction[0][id]*100))

        if (id == 0):
            return "Elefant", id
        if (id == 1):
            return "Frosch", id
        if (id == 2):
            return "Lowe", id
        if (id == 3):
            return "Schmetterling", id
        if (id == 4):
            return "Sonne", id
        if (id == 5):
            return "Vogel", id

    def loadModel(self, path):
        return load_model(path)

    def resizeImageToInputSize(self, img, image_width, image_hight):
        # resize image to match input size
        image = cv2.resize(img, (image_width, image_hight))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

if __name__ == '__main__':
    Classifire().Classifier()