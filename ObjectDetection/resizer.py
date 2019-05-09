import glob
import os

from PIL import Image
from sklearn.externals.joblib._multiprocessing_helpers import mp


def resize(f, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = f.split('/')[-1]

    try:
        # Open the image file.
        img = Image.open(f)

        # Resize it.
        img = img.resize((width, height), Image.ANTIALIAS)

        # Save it back to disk.
        img.save(os.path.join(output_path, filename))
    except:
        print("file " + str(f) + " not found")


width, height = 300, 300

input_path_elefant = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images/Elefant"
input_path_giraffe = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images/Giraffe"
input_path_kamel = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images/Kamel"
input_path_krokodil = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images/Krokodil"
input_path_loewe = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images/Loewe"
input_path_nilpferd = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images/Nilpferd"
input_path_zebra = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images/Zebra"

output_path_elefant = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images_" + str(width) + "x" + str(
    height) + "/Elefant/"
output_path_giraffe = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images_" + str(width) + "x" + str(
    height) + "/Giraffe/"
output_path_kamel = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images_" + str(width) + "x" + str(
    height) + "/Kamel/"
output_path_krokodil = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images_" + str(width) + "x" + str(
    height) + "/Krokodil/"
output_path_loewe = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images_" + str(width) + "x" + str(
    height) + "/Loewe/"
output_path_nilpferd = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images_" + str(width) + "x" + str(
    height) + "/Nilpferd/"
output_path_zebra = "/home/resi/PycharmProjects/RoboPuzzler/ObjectDetection/train_images_" + str(width) + "x" + str(
    height) + "/Zebra/"

files_elefant = glob.glob(os.path.join(input_path_elefant, "*.jpg"))
files_giraffe = glob.glob(os.path.join(input_path_giraffe, "*.jpg"))
files_kamel = glob.glob(os.path.join(input_path_kamel, "*.jpg"))
files_krokodil = glob.glob(os.path.join(input_path_krokodil, "*.jpg"))
files_loewe = glob.glob(os.path.join(input_path_loewe, "*.jpg"))
files_nilpferd = glob.glob(os.path.join(input_path_nilpferd, "*.jpg"))
files_zebra = glob.glob(os.path.join(input_path_zebra, "*.jpg"))

pool = mp.Pool(mp.cpu_count())

print('1/7')
[pool.apply(resize, args=(f, output_path_elefant)) for f in files_elefant]
print('2/7')
[pool.apply(resize, args=(f, output_path_giraffe)) for f in files_giraffe]
print('3/7')
[pool.apply(resize, args=(f, output_path_kamel)) for f in files_kamel]
print('4/7')
[pool.apply(resize, args=(f, output_path_krokodil)) for f in files_krokodil]
print('5/7')
[pool.apply(resize, args=(f, output_path_loewe)) for f in files_loewe]
print('6/7')
[pool.apply(resize, args=(f, output_path_nilpferd)) for f in files_nilpferd]
print('7/7')
[pool.apply(resize, args=(f, output_path_zebra)) for f in files_zebra]
print('done')
