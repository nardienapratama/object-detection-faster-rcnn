# import the necessary packages
import os

# initialize the base path for the synthetic dataset, where the raw images, records, and experiment logs live).
BASE_PATH = "synthetic_ds"

# build the path to the annotations file
ANNOT_PATH = os.path.sep.join([BASE_PATH, "AnnotationsFile.csv"])

# build the path to the output training and testing record files,
# along with the class labels file
TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, "records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH, "records/classes.pbtxt"])

# initialize the test split size
TEST_SIZE = 0.25

# initialize the class labels dictionary
CLASSES = {"Modem": 1}