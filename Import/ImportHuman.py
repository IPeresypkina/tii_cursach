import random
import glob

random.seed(8675309)

# load filenames in shuffled human dataset
human_files = glob.glob("../data/lfw/*/*")
random.shuffle(human_files)

# print statistics about the dataset
print('Есть %d изображений людей' % len(human_files))