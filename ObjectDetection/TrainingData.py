import Augmentor

class TrainingData:
    def generateTrainingData(self,samples):
        trainPath = {"train_images/Elefant",
                    "train_images/Frosch",
                    "train_images/Lowe",
                    "train_images/Schmetterling",
                    "train_images/Sonne",
                    "train_images/Vogel"}

        validPath = {"valid_images/Elefant",
                    "valid_images/Frosch",
                    "valid_images/Lowe",
                    "valid_images/Schmetterling",
                    "valid_images/Sonne",
                    "valid_images/Vogel"}


        for path in trainPath:
            p = Augmentor.Pipeline(path)
            p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
            p.flip_left_right(probability=0.5)
            p.flip_top_bottom(probability=0.5)
            p.random_distortion(probability=1, grid_width=6, grid_height=6, magnitude=9)
            p.sample(samples, multi_threaded=True)
        pass


if __name__ == '__main__':
    TrainingData().generateTrainingData(5)

