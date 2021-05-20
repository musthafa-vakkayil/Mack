#install packages keras and tensorflow
# importing packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow

# creating model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255)

# training data set
training_set = train_datagen.flow_from_directory('DataSet/Train',
                                                 target_size = (64, 64),
                                                 batch_size = 8,
                                                 class_mode = 'binary')

# validation data set
val_set = val_datagen.flow_from_directory('DataSet/Val',
                                            target_size = (64, 64),
                                            batch_size = 8,
                                            class_mode = 'binary')

# training
model.fit_generator(training_set,
                         steps_per_epoch = 10,
                         epochs = 50,
                         validation_data = val_set,
                         validation_steps = 2)

# saving model in hdf5 - for streamlit
tensorflow.keras.models.save_model(model,'Mack.hdf5')