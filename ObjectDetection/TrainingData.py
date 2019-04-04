import Augmentor

class TrainingData:
    def generateTrainingData(self):
        p = Augmentor.Pipeline("TrainingData/Zebra")
        # Point to a directory containing ground truth data.
        # Images with the same file names will be added as ground truth data
        # and augmented in parallel to the original data.
        # Add operations to the pipeline as normal:
        p.rotate90(probability=0.5)
        p.rotate270(probability=0.5)
        p.flip_left_right(probability=0.8)
        p.flip_top_bottom(probability=0.3)
        p.crop_random(probability=1, percentage_area=0.5)
        p.resize(probability=1.0, width=120, height=120)
        p.sample(1000)
        pass


if __name__ == '__main__':
    TrainingData().generateTrainingData()

