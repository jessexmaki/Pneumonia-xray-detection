import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Construct an ImageDataGenerator object
DIRECTORY = "Covid19-dataset/train"
BATCH_SIZE = 32

training_data_generator = ImageDataGenerator(rescale=1.0/255,
                                             zoom_range=0.1, 
                                             rotation_range=25, 
                                             width_shift_range=0.05, 
                                             height_shift_range=0.05)

validation_data_generator = ImageDataGenerator()

training_iterator = training_data_generator.flow_from_directory(DIRECTORY, class_mode='categorical', color_mode='grayscale', batch_size=BATCH_SIZE)
validation_iterator = validation_data_generator.flow_from_directory(DIRECTORY, class_mode='categorical', color_mode='grayscale', batch_size=BATCH_SIZE)

# Fixed hyperparameters
num_filters = 32  
kernel_size = 3   
dropout_rate = 0.3 
learning_rate = 0.001  

# Modify design_model function
def design_model(training_data):
    model = Sequential()
    model.add(tf.keras.Input(shape=(256, 256, 1)))
    model.add(layers.Conv2D(num_filters, (kernel_size, kernel_size), strides=3, activation="relu")) 
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Conv2D(num_filters, (kernel_size, kernel_size), strides=1, activation="relu")) 
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation="softmax"))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model

# Build and train the model with the fixed hyperparameters
model = design_model(training_iterator)

print("\nTraining model with fixed hyperparameters...")
history = model.fit(training_iterator, 
                    steps_per_epoch=training_iterator.samples/BATCH_SIZE, 
                    epochs=30,  
                    validation_data=validation_iterator, 
                    validation_steps=validation_iterator.samples/BATCH_SIZE)


# Plotting accuracy and AUC
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(['Train', 'Validation'], loc='upper left')

# Check the keys in history.history to make sure of the correct AUC key names
print(history.history.keys())

# Replace 'auc' and 'val_auc' with the correct keys if they are different
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])  
ax2.plot(history.history['val_auc'])  
ax2.set_title('Model AUC')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('AUC')
ax2.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Evaluate model on the validation set
test_steps_per_epoch = np.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Calculate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
