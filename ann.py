import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras import Sequential
from keras import layers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv('Dataset-of-Diabetes.csv')

# Check for missing values
print(dataset.isnull().sum())

# Preprocess the dataset
dataset['CLASS'] = dataset['CLASS'].replace({'N ': 'N', 'Y ': 'Y'})
dataset['Gender'] = dataset['Gender'].replace({'f': 'F'})

# Remove duplicates if any
dataset = dataset.drop_duplicates()

# Select features and target
X = pd.DataFrame(dataset.iloc[:, 2:13].values)
y = dataset.iloc[:, 13].values

# Encode categorical data
labelencoder_X_0 = LabelEncoder()
X.iloc[:, 0] = labelencoder_X_0.fit_transform(X.iloc[:, 0])

y_encoder = OneHotEncoder(categories='auto', sparse_output=False)
y = y_encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build the ANN
classifier = Sequential()

# Ensure the input shape is specified in the first layer
classifier.add(layers.Dense(units=24, kernel_initializer='uniform', activation='relu', input_shape=(X_train.shape[1],)))
classifier.add(layers.BatchNormalization())

classifier.add(layers.Dense(units=16, kernel_initializer='uniform', activation='relu'))
classifier.add(layers.BatchNormalization())

classifier.add(layers.Dense(units=3, kernel_initializer='uniform', activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()

modelPerformance = classifier.fit(x=X_train, y=y_train, batch_size=16, epochs=100, validation_split=0.3)

# Predict the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Convert predictions and true values to single labels
y_test_single = y_test.argmax(axis=1)
y_pred_single = y_pred.argmax(axis=1)

# Calculate confusion matrix and accuracy
cm = confusion_matrix(y_test_single, y_pred_single)

# Calculate and print the accuracy as a percentage
accuracy = accuracy_score(y_test_single, y_pred_single)
accuracy_percentage = accuracy * 100
print("Accuracy: {:.2f}%".format(accuracy_percentage))

# Plot confusion matrix with labels
plt.figure(figsize=(11, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Non-Diabetic', 'Predicted-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Predicted-Diabetic', 'Diabetic'])
plt.xlabel('Predicted', fontsize=16)
plt.ylabel('Factual', fontsize=16)
plt.title('Diabetes Prediction Confusion Matrix')
plt.show()

# Plotting training & validation accuracy values
plt.plot(modelPerformance.history['accuracy'])
plt.plot(modelPerformance.history['val_accuracy'])  # Include validation accuracy
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting training & validation loss values
plt.plot(modelPerformance.history['loss'])
plt.plot(modelPerformance.history['val_loss'])  # Include validation loss
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()