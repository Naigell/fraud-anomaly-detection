#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

#import CSV dataset
data = pd.read_csv('C:\\Users\\user\\Documents\\creditcard.csv')
print(data.head)

#checking the class imbalance in the dataset
print(data.groupby(['Class']).count())

#drop time column and normalize the amount column
data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1)) 

#split dataset into training and validation set
X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)
X_train = X_train.drop(['Class'], axis=1)
y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)
X_train = X_train.values
X_test = X_test.values
print(X_train.shape)

#set parameters for autoencoder
input_dim = X_train.shape[1]
encoding_dim = 24

#dense 7 layer autoencoder
'''from research, relu or its variations for the hidden layer and a linear outer layer 
   is preferred. In the process of hyperparameter tuning an elu hidden layer provided
   realtively satisfactory accuracy and was chosen.'''
model = Sequential()
model.add(Dense(encoding_dim, input_shape=(input_dim,)))
model.add(Dense(int(encoding_dim / 2), activation="elu"))
model.add(Dense(int(encoding_dim / 2), activation="elu"))
model.add(Dense(int(encoding_dim / 2), activation='elu'))
model.add(Dense(int(encoding_dim / 2), activation="elu"))
model.add(Dense(int(encoding_dim / 2), activation="elu"))
model.add(Dense(input_dim))
model.summary()

#training the model
nb_epoch = 15
batch_size = 32

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])

history = model.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    validation_data=(X_test, X_test),
                    verbose=1)
autoencoder = model

#simple plot to summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#make predictions and use them to produce reconstruction error 
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})
print(error_df.head())

threshold = 6.0
groups = error_df.groupby('true_class')
print(groups)

#plot to visualize normal and fraud transactions
fig, ax = plt.subplots(figsize=(12, 8))
for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=2.0, linestyle='',
            label = "Fraud" if name == 1 else "Normal",
            color = "red" if name == 1 else "blue")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="green", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();

normal = error_df[error_df.true_class == 0]
fraud = error_df[error_df.true_class == 1]
print('Normal transactions: %d, fraud transactions: %d' % (len(normal), len(fraud)))

#use the set threshold to determine true and false positives and negatives
true_positives = len(fraud[fraud.reconstruction_error >= threshold])
false_positives = len(normal[normal.reconstruction_error >= threshold])
true_negatives = len(normal[normal.reconstruction_error < threshold])
false_negatives = len(fraud[fraud.reconstruction_error < threshold])

print('True positives: %d, true negatives: %d' % (true_positives, true_negatives))
print('False positives: %d, false negatives: %d' % (false_positives, false_negatives))

labels = ["Normal", "Fraud"]
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)

#plot confusion matrix to visualize true and false positives and negatives
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

#save model
autoencoder.save("anomaly_model.h5")
print('saved model to disk')

