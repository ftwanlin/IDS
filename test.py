import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Convolution1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf

# X_train = pickle.load('X_train_pca.pkl')
# X_test = pickle.load("X_test_pca.pkl")
# Y_train = pickle.load('Y_train.pkl')
# Y_test = pickle.load('Y_test.pkl')

X_train = pd.read_pickle('X_train_pca.pkl')
X_test = pd.read_pickle('X_test_pca.pkl')
Y_train = pd.read_pickle('Y_train.pkl')
Y_test = pd.read_pickle('Y_test.pkl')

# X_train = tf.convert_to_tensor(X_train)
# X_test = tf.convert_to_tensor(X_test)
# Y_train = tf.convert_to_tensor(Y_train)
# Y_test = tf.convert_to_tensor(Y_test)

def create_lstm_model(input_shape, num_classes):
    model = Sequential()

    model.add(Convolution1D(64, 3, padding="same", activation="relu", input_shape=input_shape))
    model.add(Convolution1D(64, 3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Convolution1D(128, 3, padding="same", activation="relu"))
    model.add(Convolution1D(128, 3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    return model

input_shape = (122, 1)
num_classes = 5
model_lstm = create_lstm_model(input_shape, num_classes)
# define optimizer and objective, compile lstm
model_lstm.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model_lstm.summary()


# Split data: 75% training and 25% testing
train_X, test_X, train_y, test_y = train_test_split(X_train, Y_train, test_size=0.25, random_state=101)

# from sklearn.preprocessing import MinMaxScaler

# X_train = MinMaxScaler().fit_transform(X_train)
# X_test = MinMaxScaler().fit_transform(X_test)

# # x_columns_train = X_train.columns.drop('Class')
# # x_train_array = train_X[x_columns_train].values
# x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))

# dummies = pd.get_dummies(train_y)  # Classification
# outcomes = dummies.columns
# num_classes = len(outcomes)
# y_train_1 = dummies.values

# print(x_train_1.shape, y_train_1.shape)

# history = model_lstm.fit(x_train_1, y_train_1, epochs=10, batch_size=64)

# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()

# pred1 = model_lstm.predict(x_train_1)
# pred2 = np.argmax(pred1, axis=1)
# pred = le.fit_transform(pred2)
# # pred = le.inverse_transform(pred)
# y_eval = np.argmax(y_train_1, axis=1)
# score = metrics.accuracy_score(y_eval, pred)
# print("Validation score: {}%".format(score * 100))

# acc = accuracy_score(y_eval, pred)
# print("accuracy : ", acc)
# recall = recall_score(y_eval, pred, average='weighted', zero_division=0)
# print("recall : ", recall)
# precision = precision_score(y_eval, pred, average='weighted', zero_division=0)
# print("precision : ", precision)
# f1_scr = f1_score(y_eval, pred, average='weighted', zero_division=0)
# print("f1_score : ", f1_scr)

# print("####   0:Dos  1:normal  2:Probe  3:R2L  4:U2L  ###\n\n")
# print(classification_report(y_eval, pred, zero_division=0))

# cm = confusion_matrix(y_eval, pred2)
# print(cm, '\n')

# FP = cm[0, 1]
# FN = cm[1, 0]
# TP = cm[0, 0]
# TN = cm[1, 1]

# FP = FP.astype(float)
# FN = FN.astype(float)
# TP = TP.astype(float)
# TN = TN.astype(float)


# print("TP = ", TP)
# print("TN = ", TN)
# print("FP = ", FP)
# print("FN = ", FN)
# print("\n")

# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP / (TP + FP)
# print("TPR = ", TPR, "  True positive rate, Sensitivity, hit rate, or recall")
# # Fall out or false positive rate
# FPR = FP / (FP + TN)
# print("FPR = ", FPR, "  False positive rate or fall out")
# # Specificity or true negative rate
# TNR = TN / (TN + FP)
# print("TNR = ", TNR, "  True negative rate or specificity")
# # False negative rate
# FNR = FN / (TP + FN)
# print("FNR = ", FNR, "  False negative rate")
# # False Omission Rate
# FOR = FN/(FN + TN)
# print("FOR = ", FOR, "  False omission rate")

# print("\n")

# # Precision or positive predictive value
# PPV = TP / (TP + FP)
# print("PPV = ", PPV, "  Positive predictive value or precision")
# # Negative predictive value
# NPV = TN / (TN + FN)
# print("NPV = ", NPV, "  Negative predictive value")
# # False discovery rate
# FDR = FP / (TP + FP)
# print("FDR = ", FDR, "  False discovery rate")
# print("\n")

# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)
# print("ACC = ", ACC)


# ======================================================



# from sklearn.preprocessing import MinMaxScaler

# X_train = MinMaxScaler().fit_transform(X_train)
# X_test = MinMaxScaler().fit_transform(X_test)

# x_columns_train = train_df_2.columns.drop('Class')
# x_train_array = train_X[x_columns_train].values
# x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))

# dummies = pd.get_dummies(train_y)  # Classification
# outcomes = dummies.columns
# num_classes = len(outcomes)
# y_train_1 = dummies.values

x_train_1 = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))

history = model_lstm.fit(x_train_1, train_y, epochs=5, batch_size=64)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn import preprocessing

Y_pred = model_lstm.predict(X_test)
print(classification_report(Y_test, Y_pred, zero_division=0))

cm = confusion_matrix(Y_test, Y_pred)

# le = preprocessing.LabelEncoder()

# pred1 = model_lstm.predict(x_train_1)
# pred2 = np.argmax(pred1, axis=1)
# pred = le.fit_transform(pred2)
# # pred = le.inverse_transform(pred)
# y_eval = np.argmax(y_train_1, axis=1)
# score = metrics.accuracy_score(y_eval, pred)
# print("Validation score: {}%".format(score * 100))

# acc = accuracy_score(y_eval, pred)
# print("accuracy : ", acc)
# recall = recall_score(y_eval, pred, average='weighted', zero_division=0)
# print("recall : ", recall)
# precision = precision_score(y_eval, pred, average='weighted', zero_division=0)
# print("precision : ", precision)
# f1_scr = f1_score(y_eval, pred, average='weighted', zero_division=0)
# print("f1_score : ", f1_scr)

# print("####   0:Dos  1:normal  2:Probe  3:R2L  4:U2L  ###\n\n")
# print(classification_report(y_eval, pred, zero_division=0))

# cm = confusion_matrix(y_eval, pred2)
# print(cm, '\n')

FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[0, 0]
TN = cm[1, 1]

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)


print("TP = ", TP)
print("TN = ", TN)
print("FP = ", FP)
print("FN = ", FN)
print("\n")

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FP)
print("TPR = ", TPR, "  True positive rate, Sensitivity, hit rate, or recall")
# Fall out or false positive rate
FPR = FP / (FP + TN)
print("FPR = ", FPR, "  False positive rate or fall out")
# Specificity or true negative rate
TNR = TN / (TN + FP)
print("TNR = ", TNR, "  True negative rate or specificity")
# False negative rate
FNR = FN / (TP + FN)
print("FNR = ", FNR, "  False negative rate")
# False Omission Rate
FOR = FN/(FN + TN)
print("FOR = ", FOR, "  False omission rate")

print("\n")

# Precision or positive predictive value
PPV = TP / (TP + FP)
print("PPV = ", PPV, "  Positive predictive value or precision")
# Negative predictive value
NPV = TN / (TN + FN)
print("NPV = ", NPV, "  Negative predictive value")
# False discovery rate
FDR = FP / (TP + FP)
print("FDR = ", FDR, "  False discovery rate")
print("\n")

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("ACC = ", ACC)
