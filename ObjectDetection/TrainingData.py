import Augmentor

class TrainingData:
    def generateTrainingData(self,samples):
        trainPath = {"Images/train/Elefant",
                    "Images/train/Frosch",
                    "Images/train/Lowe",
                    "Images/train/Schmetterling",
                    "Images/train/Sonne",
                    "Images/train/Vogel"}

        testPath = {"Images/test/Elefant",

                    "Images/test/Frosch",
                    "Images/test/Lowe",
                    "Images/test/Schmetterling",
                    "Images/test/Sonne",
                    "Images/test/Vogel"}


        validationPath = {"Images/validate/Elefant",
                     "Images/validate/Frosch",
                     "Images/validate/Lowe",
                     "Images/validate/Schmetterling",
                     "Images/validate/Sonne",
                     "Images/validate/Vogel"}


        for path in trainPath:
            p = Augmentor.Pipeline(path)
            p.rotate(probability=0.75, max_left_rotation=2, max_right_rotation=2)
            p.flip_random(probability=0.75)
            p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=1)
            p.sample(samples, multi_threaded=True)
        pass


if __name__ == '__main__':
    TrainingData().generateTrainingData(1000)

