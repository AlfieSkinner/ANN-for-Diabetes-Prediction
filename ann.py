#Import libraries and functions
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras import Sequential
from keras import layers
from keras import regularizers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv('Dataset-of-Diabetes.csv')

# Check for missing values in the dataset
missing_values = dataset.isnull().sum()
print(missing_values)

# If there are missing values, remove rows with missing values
dataset = dataset.dropna()

# Preprocess the dataset
# Correcting inconsistencies in categorical values
dataset['CLASS'] = dataset['CLASS'].replace({'N ': 'N', 'Y ': 'Y'})
dataset['Gender'] = dataset['Gender'].replace({'f': 'F'})

# Remove duplicates if any exist
dataset = dataset.drop_duplicates()

# Select features and target
# X contains the feature variables, y contains the target variable
X = pd.DataFrame(dataset.iloc[:, 2:13].values)
y = dataset.iloc[:, 13].values

# Encode categorical data
# Label encode the first column in X turning gender values into numeric values
labelencoder_X_0 = LabelEncoder()
X.iloc[:, 0] = labelencoder_X_0.fit_transform(X.iloc[:, 0])

# One-hot encode the target variable y
y_encoder = OneHotEncoder(categories='auto', sparse_output=False)
y = y_encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature scaling to normalize the feature variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build the ANN
classifier = Sequential()

# Add the input layer and first hidden layer with batch normalization
classifier.add(layers.Dense(units=32, kernel_initializer='uniform', activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.01)))
classifier.add(layers.BatchNormalization())
classifier.add(layers.Dropout(0.3))

# Add the second hidden layer with batch normalization
classifier.add(layers.Dense(units=32, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
classifier.add(layers.BatchNormalization())
classifier.add(layers.Dropout(0.3))

# Add the output layer
classifier.add(layers.Dense(units=3, kernel_initializer='uniform', activation='softmax'))

# Compile the ANN with Adam optimizer and categorical crossentropy loss function
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()

# Fit the ANN to the training set with a validation split of 0.3, and add early stopping to the model
modelPerformance = classifier.fit(x=X_train, y=y_train, batch_size=20, epochs=100, validation_split=0.3)

# Predict the test set results
y_pred = classifier.predict(X_test)

# Convert predictions and true values to single labels
y_test_single = y_test.argmax(axis=1)
y_pred_single = y_pred.argmax(axis=1)

# Calculate confusion matrix and accuracy
conMat = confusion_matrix(y_test_single, y_pred_single)

# Calculate and print the accuracy as a percentage
accuracy = accuracy_score(y_test_single, y_pred_single)
accuracy_percentage = accuracy * 100
print("Accuracy: {:.2f}%".format(accuracy_percentage))

# Create a plot confusion matrix
plt.figure(figsize=(11, 7))
sns.heatmap(conMat, annot=True, fmt='d', cmap='Purples', xticklabels=['Diabetic', 'Predicted-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Predicted-Diabetic', 'Diabetic'])
plt.xlabel('Predicted', fontsize=16)
plt.ylabel('Factual', fontsize=16)
plt.title('Diabetes Prediction Confusion Matrix')
plt.show()

# Plot training & validation accuracy values over epochs
plt.plot(modelPerformance.history['accuracy'])
plt.plot(modelPerformance.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Data', 'Validation Data'], loc='upper left')
plt.show()

# Plot training & validation loss values over epochs
plt.plot(modelPerformance.history['loss'])
plt.plot(modelPerformance.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Data', 'Validation Data'], loc='upper left')
plt.show()