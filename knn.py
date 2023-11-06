from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import argparse
import sys
sys.path.append("..")
from library.dataset.simpledatasetloader import SimpleDatasetLoader
from library.preprocessing.simplepreprocessor import SimplePreprocessor

# arg_p = argparse.ArgumentParser()

# arg_p.add_argument("-d", "--dataset", required = True, help = "path to input dataset")
# arg_p.add_argument("-k", "--neighbors", type = int, default = 1, help = "# of nearest neighbors for classification")
# arg_p.add_argument("-j", "--jobs", type = int, default = -1, help = "# of jobs for k-NN distance (-1 uses all avilable cores)")

# args = vars(arg_p.parse_args())

print("INFO - loading images...")

imagePaths = list(paths.list_images("dataset/animal"))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.reshape((data.shape[0], 3072))

print("INFO - features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

print("INFO - evaluating k-NN classifier...]")

model = KNeighborsClassifier(n_neighbors = 1, n_jobs = -1)
model.fit(trainX, trainY)

print(classification_report(testY, model.predict(testX), target_names = le.classes_))