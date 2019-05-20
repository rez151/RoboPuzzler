from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time as time

from keras import backend as K
import matplotlib.pyplot as plt

from ObjectDetection.TensorBoardWrapper import TensorBoardWrapper


class TrainModel:
        def trainModel(self):
                K.set_image_dim_ordering('tf')

                trainPath = 'Images/train'
                validationPath = 'Images/validate'

                image_width = 224
                image_hight = 224
                batch_size = 10

                model = Sequential()
                model.add(Conv2D(32, (3, 3), input_shape=(image_width, image_hight, 1)))
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
                model.add(Activation('relu'))
                model.add(Dropout(0.5))
                model.add(Dense(6))
                model.add(Activation('softmax'))


                #callbacks
                callback_list = [
                         EarlyStopping(
                                 monitor='acc',
                                 patience=1,
                         ),
                         ModelCheckpoint(
                                 filepath='model/first_try.h5',
                                 monitor='val_loss',
                                 save_best_only=True
                         ),
                         ReduceLROnPlateau(
                                 monitor='val_loss',
                                 factor=0.1,
                                 patience=3
                         )
                ]


                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                train_datagen = ImageDataGenerator(rescale=1. / 255,)

                test_datagen = ImageDataGenerator(rescale=1. / 255)

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

                #callbacks = [
                #        TensorBoardWrapper(
                #                validation_generator,
                #                nb_steps=5,
                #                log_dir='my_log_dir/{}'.format(time()),
                #                histogram_freq=1,
                #                batch_size=32,
                #                write_graph=True,
                #                write_grads=True)]


                try:
                        history = model.fit_generator(
                                train_generator,
                                callbacks=callback_list,
                                epochs=20,
                                steps_per_epoch=6000/batch_size,
                                validation_data=validation_generator,
                                validation_steps=2466/batch_size,
                                verbose=1
                                )

                        # summarize history for accuracy
                        plt.plot(history.history['acc'])
                        plt.plot(history.history['val_acc'])
                        plt.title('model accuracy')
                        plt.ylabel('accuracy')
                        plt.xlabel('epoch')
                        plt.legend(['train', 'test'], loc='upper left')
                        plt.show()
                        plt.savefig('logs/history_for_accuracy{}.jpg'.format(time()))
                        # summarize history for loss
                        plt.plot(history.history['loss'])
                        plt.plot(history.history['val_loss'])
                        plt.title('model loss')
                        plt.ylabel('loss')
                        plt.xlabel('epoch')
                        plt.legend(['train', 'test'], loc='upper left')
                        plt.show()
                        plt.savefig('logs/history_for_loss{}.jpg'.format(time()))

                        model.save_weights('model/first_try.h5')
                except KeyboardInterrupt:
                        model.save_weights('model/first_try.h5')


if __name__ == '__main__':
        TrainModel().trainModel()
