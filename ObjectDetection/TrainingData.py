import Augmentor
import tensorflow as tf




class TrainingData:
    def generateTrainingData(self):
        p = Augmentor.Pipeline("TrainingData/Zebra")
        # Point to a directory containing ground truth data.
        # Images with the same file names will be added as ground truth data
        # and augmented in parallel to the original data.
        # Add operations to the pipeline as normal:
        p.ground_truth("/ObjectDetection/TrainingData/Zebra")
        p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.flip_top_bottom(probability=0.5)
        p.sample(50)
        pass


if __name__ == '__main__':
    TrainingData().generateTrainingData()

