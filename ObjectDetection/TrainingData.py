import Augmentor

class TrainingData:
    def generateTrainingData(self,samples):
        filePath = {"TrainingData/Elefant",
                    "TrainingData/Giraffe",
                    "TrainingData/Kamel",
                    "TrainingData/Krokodil",
                    "TrainingData/Lion",
                    "TrainingData/Nilpferd",
                    "TrainingData/Zebra"}

        for path in filePath:
            p = Augmentor.Pipeline(path)
            p.rotate(probability=0.5,max_left_rotation=10,max_right_rotation=10)
            p.flip_random(probability=0.5)
            p.random_distortion(probability=0.15, grid_width=20, grid_height=20, magnitude=1)
            p.skew(probability=0.5,magnitude=0.1)
            p.sample(samples)
        pass


if __name__ == '__main__':
    TrainingData().generateTrainingData(100)

