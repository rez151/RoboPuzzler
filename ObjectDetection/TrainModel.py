from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras

class TrainModel:
        def trainModel(self):
                K.set_image_dim_ordering('tf')

                trainPath = 'Images/train'
                validationPath = 'Images/validate'

                image_width = 150
                image_hight = 150
                batch_size = 10

                model = Sequential()
                model.add(Conv2D(32, (3, 3), input_shape=(image_width, image_hight, 1)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Conv2D(32, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

                model.add(Conv2D(64, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
                model.add(Dropout(0.1))

                model.add(Flatten())
                model.add(Dense(64))
                model.add(Activation('relu'))
                model.add(Dropout(0.5))
                model.add(Dense(6))
                model.add(Activation('softmax'))


                #callbacks
                callback_list = [
                        keras.callbacks.EarlyStopping(
                                monitor='acc',
                                patience=1,
                        ),
                        keras.callbacks.ModelCheckpoint(
                                filepath='model/first_try.h5',
                                monitor='val_loss',
                                save_best_only=True
                        ),
                        keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.1,
                                patience=1
                        )
                ]

                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                train_datagen = ImageDataGenerator(rescale=1. / 255,)

                test_datagen = ImageDataGenerator(rescale=1. / 224)

                train_generator = train_datagen.flow_from_directory(
                        trainPath,
                        color_mode='grayscale',
                        target_size=(image_width, image_hight),
                        batch_size=batch_size,
                        class_mode='categorical')

                validation_generator = test_datagen.flow_from_directory(
                        validationPath,
                        color_mode='grayscale',
                        target_size=(image_width, image_hight),
                        batch_size=batch_size,
                        class_mode='categorical')



                try:
                        model.fit_generator(
                                train_generator,
                                callbacks=callback_list,
                                epochs=20,
                                steps_per_epoch= 6000/batch_size,
                                validation_data=validation_generator,
                                validation_steps=2466/batch_size
                                )
                        model.save_weights('model/first_try.h5')
                except KeyboardInterrupt:
                        model.save_weights('model/first_try.h5')


if __name__ == '__main__':
        TrainModel().trainModel()
