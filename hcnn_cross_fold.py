import tensorflow as tf
import numpy as np
import os
import numpy as np
import cv2
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2

#read an image and resize it to (128, 128,1)
# Set input and output directories
input_dir = r'D:\Github\qnn-qhw\Combined_data_resized'
output_dir = r'D:\Github\qnn-qhw'

# Set target size
target_size = (256, 256)

# Get list of image files in input directory
files = glob.glob(os.path.join(input_dir, '*.jpg'))

# Loop over files in batches and resize
batch_size = 89
for i in range(0, len(files), batch_size):
    batch_files = files[i:i+batch_size]
    batch_images = []
    for file in batch_files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, target_size)
        batch_images.append(image.reshape((*target_size, 1)))
    batch_images = np.array(batch_images)
    output_path = os.path.join(output_dir, f'batch_{i}_oral.npy')
    np.save(output_path, batch_images)


# Load data
oral_data = np.load(r"D:\Github\qnn-qhw\conv_batch_0_oral_1_20_imgs.npy")
nonoral_data = np.load(r"D:\Github\qnn-qhw\conv_batch_89_non_oral_1_80_imgs.npy")


# Set labels
oral_labels = np.ones(oral_data.shape[0], dtype=np.int32)*0
nonoral_labels = np.ones(nonoral_data.shape[0], dtype=np.int32)*1


all_data = np.concatenate([oral_data,nonoral_data], axis=0)
all_labels = np.concatenate([oral_labels, nonoral_labels])

# One-hot encode labels
one_hot_label = to_categorical(all_labels)

# Initialize k-fold cross-validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize list to store Test Accuracies
test_accs = []

# Initialize list to store Test Losses
test_losses = []


# # Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(4, (2, 2), activation='relu', kernel_initializer=GlorotUniform(),
                           input_shape=(128, 128, 4)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=1),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dense(2, activation='softmax', kernel_initializer=GlorotUniform())
])

# Train and evaluate model using Stratified K Fold cross-validation

y_true_all = []
y_pred_all = []
for train_index, test_index in skf.split(all_data, all_labels):
    train_data, test_data = all_data[train_index], all_data[test_index]
    train_labels, test_labels = one_hot_label[train_index], one_hot_label[test_index]
    
    # Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=10e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_data, train_labels, batch_size=10, epochs=20)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data,test_labels, verbose=1)
    print('Test accuracy:', test_acc)

    # Append test accuracy to list
    test_accs.append(test_acc)
    
    # Predict the test labels
    y_pred = np.argmax(model.predict(test_data), axis=-1)
    y_true = test_labels
    
    # Append true and predicted labels to lists
    y_true_all.extend(y_true)
    y_pred_all.extend(y_pred)

    test_losses.append(history.history['loss'][-1]) # append the final loss value of the last epoch to the list
    
    
