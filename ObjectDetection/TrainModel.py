from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from time import time
import keras

from ObjectDetection.TensorBoardWrapper import TensorBoardWrapper


class TrainModel:
        def trainModel(self):
                K.set_image_dim_ordering('tf')

                trainPath = 'Images/train'
                validationPath = 'Images/validate'

                image_width = 224
                image_hight = 224
                batch_size = 10

                model = keras.models.Sequential()
                model.add(Conv2D(32, (3, 3), input_shape=(image_width, image_hight, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Conv2D(32, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Conv2D(64, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.1))

                model.add(Flatten())
                model.add(Dense(64))
                model.add(Activation('relu'))
                model.add(Dropout(0.5))
                model.add(Dense(6))
                model.add(Activation('softmax'))


                #callbacks
                # callback_list = [
                #         keras.callbacks.EarlyStopping(
                #                 monitor='acc',
                #                 patience=1,
                #         ),
                #         keras.callbacks.ModelCheckpoint(
                #                 filepath='model/first_try.h5',
                #                 monitor='val_loss',
                #                 save_best_only=True
                #         ),
                #         keras.callbacks.ReduceLROnPlateau(
                #                 monitor='val_loss',
                #                 factor=0.1,
                #                 patience=1
                #         )
                # ]


                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                train_datagen = ImageDataGenerator(rescale=1. / 255,)

                test_datagen = ImageDataGenerator(rescale=1. / 255)

                train_generator = train_datagen.flow_from_directory(
                        trainPath,
                        color_mode='rgb',
                        target_size=(image_width, image_hight),
                        batch_size=batch_size,
                        class_mode='categorical')

                validation_generator = test_datagen.flow_from_directory(
                        validationPath,
                        color_mode='rgb',
                        target_size=(image_width, image_hight),
                        batch_size=batch_size,
                        class_mode='categorical')

                callbacks = [
                        TensorBoardWrapper(
                                validation_generator,
                                nb_steps=5,
                                log_dir='my_log_dir/{}'.format(time()),
                                histogram_freq=1,
                                batch_size=32,
                                write_graph=False,
                                write_grads=True)]

                try:
                        model.fit_generator(
                                train_generator,
                                #callbacks=callback_list,
                                callbacks=callbacks,
                                epochs=1,
                                steps_per_epoch= 6000/batch_size,
                                validation_data=validation_generator,
                                validation_steps=2466/batch_size
                                )
                        model.save_weights('model/first_try.h5')
                except KeyboardInterrupt:
                        model.save_weights('model/first_try.h5')


if __name__ == '__main__':
        TrainModel().trainModel()
