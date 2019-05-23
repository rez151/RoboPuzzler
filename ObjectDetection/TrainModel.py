from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
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


        # Defines the network to use only tensorflow
                K.set_image_dim_ordering('tf')
        # Ordnerstruktur
                trainPath = 'Images/train'
                test_dir = 'Images/test'
                validationPath = 'Images/validate'

        #Bilderformat
                colerType = 1
                image_width = 224
                image_hight = 224
        # batch_size, defines the number of samples that will be propagated through the network
                batch_size = 10
        # Sequential, prepare the model to a list of layers
                model = Sequential()
        # Definition for the layout of the network
                dense_layers = [0]
                layer_sizes = [32, 32, 64, 64]
                conv_layers = [4]
        # Coonv2D(number of filters(window_height, window_width), Activationfunction, input_shape)
        # MaxPooling2D(pool_size=(2, 2))
        # loop for network layer
                for dense_layer in dense_layers:
                        for conv_layer in conv_layers:
                                i = 0
                                model.add(Conv2D(
                                        layer_sizes[0],
                                        (3, 3),
                                        input_shape=(
                                                image_width,
                                                image_hight,
                                                colerType
                                        )
                                ))
                                model.add(Activation('relu'))
                                model.add(MaxPooling2D(pool_size=(2, 2)))

                                for l in range(conv_layer - 1):
                                        model.add(Conv2D(layer_sizes[i + 1], (3, 3)))
                                        model.add(Activation('relu'))
                                        model.add(MaxPooling2D(pool_size=(2, 2)))
                                        i += 1

                                model.add(Flatten())
                                for _ in range(dense_layer):
                                        model.add(Dense())
                                        model.add(Activation('relu'))

                                model.add(Dropout(0.5))
                                model.add(Dense(6))
                                model.add(Activation('softmax'))

                model.compile(loss='categorical_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy'])


                train_datagen = ImageDataGenerator(rescale=1. / 255, )

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

                try:
                        history = model.fit_generator(
                                train_generator,
                                #callbacks=callback_list,
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
                                        layer_sizes[2],
                                        layer_sizes[3]
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
                                        layer_sizes[2],
                                        layer_sizes[3]
                                )
                        )
                        plt.show()

                        model.summary()

                        model.save_weights('model/first_try.h5')

                except KeyboardInterrupt:
                        model.save_weights('model/first_try.h5')

if __name__ == '__main__':
 TrainModel().trainModel()

# class TrainModel:
#         def trainModel(self):
#                 K.set_image_dim_ordering('tf')
#
#                 trainPath = 'Images/train'
#                 validationPath = 'Images/validate'
#
#                 image_width = 224
#                 image_hight = 224
#                 batch_size = 10
#
#                 model = Sequential()
#                 model.add(Conv2D(32, (3, 3), input_shape=(image_width, image_hight, 1)))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))
#
#                 model.add(Conv2D(32, (3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))
#
#                 model.add(Conv2D(64, (3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))
#
#                 model.add(Flatten())
#                 model.add(Activation('relu'))
#                 model.add(Dropout(0.5))
#                 model.add(Dense(6))
#                 model.add(Activation('softmax'))
#
#
#                 #callbacks
#                 callback_list = [
#                          EarlyStopping(
#                                  monitor='acc',
#                                  patience=1,
#                          ),
#                          ModelCheckpoint(
#                                  filepath='model/first_try.h5',
#                                  monitor='val_loss',
#                                  save_best_only=True
#                          ),
#                          ReduceLROnPlateau(
#                                  monitor='val_loss',
#                                  factor=0.1,
#                                  patience=3
#                          )
#                 ]
#
#
#                 model.compile(loss='categorical_crossentropy',
#                               optimizer='adam',
#                               metrics=['accuracy'])
#
#                 train_datagen = ImageDataGenerator(rescale=1. / 255,)
#
#                 test_datagen = ImageDataGenerator(rescale=1. / 255)
#
#                 train_generator = train_datagen.flow_from_directory(
#                         trainPath,
#                         color_mode='grayscale',
#                         target_size=(image_width, image_hight),
#                         batch_size=batch_size,
#                         class_mode='categorical')
#
#                 validation_generator = test_datagen.flow_from_directory(
#                         validationPath,
#                         color_mode='grayscale',
#                         target_size=(image_width, image_hight),
#                         batch_size=batch_size,
#                         class_mode='categorical')
#
#                 #callbacks = [
#                 #        TensorBoardWrapper(
#                 #                validation_generator,
#                 #                nb_steps=5,
#                 #                log_dir='my_log_dir/{}'.format(time()),
#                 #                histogram_freq=1,
#                 #                batch_size=32,
#                 #                write_graph=True,
#                 #                write_grads=True)]
#
#
#                 try:
#                         history = model.fit_generator(
#                                 train_generator,
#                                 callbacks=callback_list,
#                                 epochs=20,
#                                 steps_per_epoch=6000/batch_size,
#                                 validation_data=validation_generator,
#                                 validation_steps=2466/batch_size,
#                                 verbose=1
#                                 )
#
#                         # summarize history for accuracy
#                         plt.plot(history.history['acc'])
#                         plt.plot(history.history['val_acc'])
#                         plt.title('model accuracy')
#                         plt.ylabel('accuracy')
#                         plt.xlabel('epoch')
#                         plt.legend(['train', 'test'], loc='upper left')
#                         plt.show()
#                         plt.savefig('logs/history_for_accuracy{}.jpg'.format(time.time()))
#                         # summarize history for loss
#                         plt.plot(history.history['loss'])
#                         plt.plot(history.history['val_loss'])
#                         plt.title('model loss')
#                         plt.ylabel('loss')
#                         plt.xlabel('epoch')
#                         plt.legend(['train', 'test'], loc='upper left')
#                         plt.show()
#                         plt.savefig('logs/history_for_loss{}.jpg'.format(time.time()))
#
#                         model.save_weights('model/first_try.h5')
#                 except KeyboardInterrupt:
#                         model.save_weights('model/first_try.h5')
#
#
# if __name__ == '__main__':
#         TrainModel().trainModel()