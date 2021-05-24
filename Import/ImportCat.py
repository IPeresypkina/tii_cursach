import random
import glob

random.seed(8675309)

# load filenames in shuffled human dataset
cat_files = glob.glob("../data/cat-breeds-dataset/images/*/*")
random.shuffle(cat_files)

# print statistics about the dataset
print('Есть %d изображений котов' % len(cat_files))