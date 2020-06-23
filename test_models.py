from tensorflow.keras.models import load_model


from preprocess import audio2vector,audio2Image
import csv
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

model = load_model('model.h5',compile=False)
model.load_weights('model_weight.h5')
model2 = load_model('model2.h5',compile=False)
model2.load_weights('model_weight2.h5')

#model._make_predict_function()
model2._make_predict_function()

import enum
print(enum.__file__)

reader = csv.reader(open('dictonary.csv', 'r'))
labels_dict = {}
for row in reader:
    k, v = row
    labels_dict[k] = v


num_rows = 64
num_columns = 64
num_channels = 3


f='speech/Natalia/h2.wav'
audio2Image('Tal',f,6)

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('./data/val',target_size=(64, 64),batch_size=32,class_mode='categorical',shuffle = False)

#file = audio2vector(f)
#file = file.reshape(1, num_rows, num_columns, num_channels)




#speaker


prediction = model2.predict(training_set)
print(prediction)
filenames=training_set.filenames

pred = model2.predict_generator(training_set, steps=50, verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
print(predicted_class_indices)
labels = (training_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions = predictions[:len(filenames)]
print(filenames)
print(predictions)


#speech

reader = csv.reader(open('speech.csv', 'r'))
labels_dict2 = {}
for row in reader:
    k, v = row
    labels_dict2[k] = v


file=audio2vector('speech/Eli/ar-21.wav')
file = file.reshape(1, 10, 4, 1)


item_prediction = model.predict_classes(file)
for i, j in labels_dict2.items():
    if j == str(item_prediction[0]):
        pred=i
        print('is keyword? {}'.format(i))

"""


prediction = model2.predict(file)
item_prediction = model2.predict_classes(file)
for i, j in labels_dict.items():
    if j == str(item_prediction[0]):
        print('speaker: {}'.format(i))

"""