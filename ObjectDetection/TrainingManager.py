import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.layers.core import Lambda


from keras import backend as K

import time as time


import matplotlib.pyplot as plt

from ObjectDetection.TensorBoardWrapper import TensorBoardWrapper


class TrainModel:
        def trainModel(self):
        ## required for efficient GPU use
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            session = tf.Session(config=config)
            K.set_session(session)
        ## required for efficient GPU use

        #TODO Training on multi GPU

        # Defines the network to use only tensorflow
            K.set_image_dim_ordering('tf')
        # Ordnerstruktur
            trainPath = 'Images/train'
            test_dir = 'Images/test'
            validationPath = 'Images/validate'

        #Bilderformat
            colermode = 1
            image_width = 224
            image_hight = 224
        # batch_size, defines the number of samples that will be propagated through the network
            batch_size = 10
        # Sequential, prepare the model to a list of layers
            model = Sequential()
        # Definition for the layout of the network
            dense_layers = [1]
            layer_sizes = [32, 32, 64]
            conv_layers = [3]
        # Coonv2D(number of filters(window_height, window_width), Activationfunction, input_shape)
        # MaxPooling2D(pool_size=(2, 2))
        # loop for network layer

            model.add(Conv2D(32, (3, 3), padding='same', input_shape=(image_width, image_hight,colermode)))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (3, 3), padding='same'))
            model.add(Activation('relu'))
            # The next layer is the substitute of max pooling, we are taking a strided convolution layer to reduce the dimensionality of the image.
            model.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(32, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (3, 3), padding='same'))
            model.add(Activation('relu'))
            # The next layer is the substitute of max pooling, we are taking a strided convolution layer to reduce the dimensionality of the image.
            model.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (1, 1), padding='valid'))
            model.add(Activation('relu'))
            model.add(Conv2D(6, (1, 1), padding='valid'))
            model.add(GlobalAveragePooling2D())
            model.add(Activation('softmax'))
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                            featurewise_center=False,  # set input mean to 0 over the dataset
                                            samplewise_center=False,  # set each sample mean to 0
                                            featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                            samplewise_std_normalization=False,  # divide each input by its std
                                            zca_whitening=False,  # apply ZCA whitening
                                            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                                            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                            horizontal_flip=True,  # randomly flip images
                                            vertical_flip=False )

            test_datagen = ImageDataGenerator(rescale=1. / 255)

            callbacks_list = [
                ModelCheckpoint("model/weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True,
                                         save_weights_only=False, mode='max')]

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
                    history = model.fit_generator(
                            train_generator,
                            callbacks=callbacks_list,
                            epochs=20,
                            steps_per_epoch=6000/batch_size,
                            validation_data=validation_generator,
                            validation_steps=2466/batch_size,
                            verbose=1
                            )

                    # generate diagrams Klassifizierung and Verlustfunktion
                    acc = history.history['acc']
                    val_acc = history.history['val_acc']
                    loss = history.history['loss']
                    val_loss = history.history['val_loss']
                    epochs = range(1, len(acc) + 1)
                    plt.figure()
                    plt.plot(epochs, acc, 'bo', label='Training')
                    plt.plot(epochs, val_acc, 'b', label='Validierung')
                    plt.title('Korrektklassifizierungsrate Training/Validierung')
                    plt.legend()
                    plt.savefig(
                            'log_img/Klassifizierung_ConvLayers_{}_layersizes_{}_{}_{}.png'.format(
                                    conv_layers[0],
                                    layer_sizes[0],
                                    layer_sizes[1],
                                    layer_sizes[2]
                            )
                    )
                    plt.figure()
                    plt.plot(epochs, loss, 'bo', label='Verlust Training')
                    plt.plot(epochs, val_loss, 'b', label='Verlust Validierung')
                    plt.title('Wert der Verlustfunktion Training/Validierung')
                    plt.legend()
                    plt.savefig(
                            'log_img//Verlustfunktion_ConvLayers_{}_layersizes_{}_{}_{}.png'.format(
                                    conv_layers[0],
                                    layer_sizes[0],
                                    layer_sizes[1],
                                    layer_sizes[2]
                            )
                    )
                    plt.show()

                    model.summary()

                    model.save_weights('model/first_try.h5')

            except KeyboardInterrupt:
                    model.save_weights('model/first_try.h5')

if __name__ == '__main__':
 TrainModel().trainModel()
