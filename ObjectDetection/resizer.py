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


width, height = 224, 224

input_path_elefant = "Images/test/Elefant/"
input_path_frosch = "Images/test/Frosch/"
input_path_lowe = "Images/test/Lowe/"
input_path_schmetterling = "Images/test/Schmetterling/"
input_path_sonne = "Images/test/Sonne/"
input_path_vogel = "Images/test/Vogel/"


output_path_elefant = "Images/test/" + str(width) + "x" + str(height) + "/Elefant/"
output_path_giraffe = "Images/test/" + str(width) + "x" + str(height) + "/Frosch/"
output_path_kamel = "Images/test/" + str(width) + "x" + str(height) + "/Lowe/"
output_path_krokodil = "Images/test/" + str(width) + "x" + str(height) + "/Schmetterling/"
output_path_loewe = "Images/test/" + str(width) + "x" + str(height) + "/Sonne/"
output_path_nilpferd = "Images/test/" + str(width) + "x" + str(height) + "/Vogel/"


files_elefant = glob.glob(os.path.join(input_path_elefant, "*.jpg"))
files_frosch = glob.glob(os.path.join(input_path_frosch, "*.jpg"))
files_lowe = glob.glob(os.path.join(input_path_lowe, "*.jpg"))
files_schmetterling = glob.glob(os.path.join(input_path_schmetterling, "*.jpg"))
files_sonne = glob.glob(os.path.join(input_path_sonne, "*.jpg"))
files_vogel = glob.glob(os.path.join(input_path_vogel, "*.jpg"))


pool = mp.Pool(mp.cpu_count())

print('1/6')
[pool.apply(resize, args=(f, output_path_elefant)) for f in files_elefant]
print('2/6')
[pool.apply(resize, args=(f, output_path_giraffe)) for f in files_frosch]
print('3/6')
[pool.apply(resize, args=(f, output_path_kamel)) for f in files_lowe]
print('4/6')
[pool.apply(resize, args=(f, output_path_krokodil)) for f in files_schmetterling]
print('5/6')
[pool.apply(resize, args=(f, output_path_loewe)) for f in files_sonne]
print('6/6')
[pool.apply(resize, args=(f, output_path_nilpferd)) for f in files_vogel]
print('done')
