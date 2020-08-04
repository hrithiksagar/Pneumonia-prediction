import os
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models  import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dropout
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
train_dir  = "train dir"
val_dir = "val dir"
test_dir = "Add test dir"

# Image augumentation :
# img = glob( train_dir + "/PNEUMONIA/*.jpeg" )
classes = ["NORMAL", "PNEUMONIA"]
train_data = glob(train_dir+"/NORMAL/*.jpeg")
train_data += glob(train_dir+"/PNEUMONIA/*.jpeg")
data_gen = ImageDataGenerator(rescale= 1./255) #Augmentation happens here

train_batches = data_gen.flow_from_directory(train_dir, target_size=(226,226), classes= classes, class_mode="categorical")
test_batches = data_gen.flow_from_directory(test_dir, target_size=(226,226), classes= classes, class_mode="categorical")
val_batches = data_gen.flow_from_directory(val_dir, target_size = (226, 226), classes = classes, class_mode = "categorical")
# conv network start:

model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=train_batches.image_shape, activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=2, activation="softmax"))
early_stop_monitor = EarlyStopping(patience=3, monitor="val_acc", mode= "max", verbose=2)
optimizer = Adam()
early_stopping_monitor = EarlyStopping(patience = 3, monitor = "val_acc", mode="max", verbose = 2)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
history = model.fit_generator(epochs=5, callbacks=[early_stopping_monitor], shuffle=True, validation_data=val_batches, generator=train_batches, steps_per_epoch=500, validation_steps=10,verbose=2)
prediction = model.predict_generator(generator=train_batches, verbose=2, steps=100)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
