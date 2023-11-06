import os

base_path = "dataset/lisa"
annot_path = os.path.sep.join([base_path, "allAnnotations.csv"])
train_path = os.path.sep.join([base_path, "/training.record"])
test_path = os.path.sep.join([base_path, "/testing.record"])
classes_file = os.path.sep.join([base_path, "/classes.pbtxt"])
test_percentage = 0.25
classes = { "pedestrianCrossing": 1, "signalAhead": 2, "stop": 3 }