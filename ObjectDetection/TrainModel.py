from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class TrainModel:
        def TrainModel(self):
                batch_size = 16
                K.set_image_dim_ordering('tf')

                model = Sequential()
                model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150,150,3)))
                model.add(Conv2D(32, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

                model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

                model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

                model.add(Flatten())
                model.add(Dense(512, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(7, activation='softmax'))

                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                train_datagen = ImageDataGenerator(
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        rescale=1./255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True)

                test_datagen = ImageDataGenerator(rescale=1./255)
                train_generator = train_datagen.flow_from_directory(
                        'train_images',
                        target_size=(150, 150),
                        batch_size=batch_size,
                        class_mode='categorical')

                validation_generator = test_datagen.flow_from_directory(
                        'train_images',
                        target_size=(150, 150),
                        batch_size=batch_size,
                        class_mode='categorical')

                model.fit_generator(
                        train_generator,
                        steps_per_epoch=2000 // batch_size,
                        epochs=40,
                        validation_data=validation_generator,
                        validation_steps=800 // batch_size)
                model.save_weights('model/first_train.h5')


        def Retrain(self):
                batch_size = 16
                K.set_image_dim_ordering('tf')
                model = Sequential()
                model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)))
                model.add(Conv2D(32, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

                model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

                model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

                model.add(Flatten())
                model.add(Dense(512, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(7, activation='softmax'))

                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                model.load_weights('model/first_train.h5')

                train_datagen = ImageDataGenerator(
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        rescale=1. / 255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True)

                test_datagen = ImageDataGenerator(rescale=1. / 255)
                train_generator = train_datagen.flow_from_directory(
                        'train_images',
                        target_size=(150, 150),
                        batch_size=batch_size,
                        class_mode='categorical')

                validation_generator = test_datagen.flow_from_directory(
                        'train_images',
                        target_size=(150, 150),
                        batch_size=batch_size,
                        class_mode='categorical')

                model.fit_generator(
                        train_generator,
                        steps_per_epoch=1500 // batch_size,
                        epochs=20,
                        validation_data=validation_generator,
                        validation_steps=800 // batch_size)
                model.save_weights('model/first_train.h5')

if __name__ == '__main__':
    TrainModel().TrainModel()