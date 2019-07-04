from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt


class TrainModel:
    @staticmethod
    def trainModel():
        ## required for efficient GPU use
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        session = tf.Session(config=config)
        K.set_session(session)

        # Defines the network to use only tensorflow
        K.set_image_dim_ordering('tf')
        # Ordnerstruktur
        trainPath = 'Images/train'
        test_dir = 'Images/test'
        validationPath = 'Images/validate'

        # Bilderformat
        colerType = 1
        image_width = 224
        image_hight = 224
        # batch_size, defines the number of samples that will be propagated through the network
        batch_size = 32
        # Sequential, prepare the model to a list of layers
        model = Sequential()
        # Definition for the layout of the network
        dense_layers = [1]
        layer_sizes = [32, 32, 64]
        conv_layers = [3]
        # Coonv2D, convolutional operation
        # MaxPooling2D(pool_size=(2, 2))
        # loop for network layer
        for dense_layer in dense_layers:
            for conv_layer in conv_layers:
                i = 0
                model.add(Conv2D(
                    layer_sizes[0],
                    (3, 3),
                    activation='relu',
                    input_shape=(
                        image_width,
                        image_hight,
                        colerType
                    )
                ))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer - 1):
                    model.add(Conv2D(layer_sizes[i + 1], (3, 3), activation='relu'))
                    model.add(BatchNormalization())
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    i += 1

                model.add(Flatten())
                for _ in range(dense_layer):
                    model.add(Dense(64, activation='relu'))
                    model.add(BatchNormalization())

                model.add(Dropout(0.5))
                model.add(Dense(6, activation='softmax'))
                model.add(BatchNormalization())

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        train_datagen = ImageDataGenerator(rescale=1. / 255)

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

        callback_list = [
            EarlyStopping(
                monitor='acc',
                patience=1
            ),
            ModelCheckpoint(
                filepath='model/first_try.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10
            )
        ]

        try:
            history = model.fit_generator(
                train_generator,
                callbacks=callback_list,
                epochs=30,
                steps_per_epoch=6138 / batch_size,
                validation_data=validation_generator,
                validation_steps=1236 / batch_size,
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
                'log_img/Verlustfunktion_ConvLayers_{}_layersizes_{}_{}_{}.png'.format(
                    conv_layers[0],
                    layer_sizes[0],
                    layer_sizes[1],
                    layer_sizes[2]
                )
            )
            plt.show()

            model.summary()

            model.save("model/model.h5")

        except KeyboardInterrupt:
            model.save("model/model.h5")
        pass

if __name__ == '__main__':
    TrainModel().trainModel()
