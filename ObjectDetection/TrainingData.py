import Augmentor

class TrainingData:
    def generateTrainingData(self,samples):
        filePath = {"train_images/Elefant",
                    "train_images/Giraffe",
                    "train_images/Kamel",
                    "train_images/Krokodil",
                    "train_images/Loewe",
                    "train_images/Nilpferd",
                    "train_images/Zebra"}


        for path in filePath:
            p = Augmentor.Pipeline(path)
            p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
            p.flip_left_right(probability=0.5)
            p.zoom_random(probability=0.5, percentage_area=0.8)
            p.flip_top_bottom(probability=0.5)
            p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
            p.sample(samples)
        pass


if __name__ == '__main__':
    TrainingData().generateTrainingData(1000)

