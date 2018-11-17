#!/usr/bin/env python3

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.optimizers import Adam
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

json_file = open("/tmp/models/propaganda_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("/tmp/models/propaganda_model.h5")

loaded_model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])

test_dir = '/tmp/test_script/images/'

img_width, img_height = 32, 32
batch_size = 16
nb_test_samples = 1

datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

scores = loaded_model.evaluate_generator(test_generator, nb_test_samples)
print("accuracy: %.2f%%" % (scores[1]*100))