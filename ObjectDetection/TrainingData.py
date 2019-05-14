import Augmentor

class TrainingData:
    def generateTrainingData(self,samples):
        trainPath = {"Images/train_images/Elefant",
                    "Images/train_images/Frosch",
                    "Images/train_images/Lowe",
                    "Images/train_images/Schmetterling",
                    "Images/train_images/Sonne",
                    "Images/train_images/Vogel"}

        testPath = {"Images/test_images/Elefant",
                     "Images/test_images/Schmetterling ",
                     "Images/test_images/Vogel"}

        validationPath = {"Images/validation_images/Elefant",
                     "Images/validation_images/Frosch",
                     "Images/validation_images/Lowe",
                     "Images/validation_images/Schmetterling",
                     "Images/validation_images/Sonne",
                     "Images/validation_images/Vogel"}


        for path in testPath:
            p = Augmentor.Pipeline(path)
            p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
            p.flip_left_right(probability=0.5)
            p.flip_top_bottom(probability=0.5)
            p.random_distortion(probability=1, grid_width=5, grid_height=5, magnitude=2)
            p.sample(samples, multi_threaded=True)
        pass


if __name__ == '__main__':
    TrainingData().generateTrainingData(1000)

