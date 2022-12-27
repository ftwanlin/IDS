import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Convolution1D, MaxPooling1D
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Loading training set into dataframe
train_df = pd.read_csv('data/KDDTrain+.csv')
# Loading testing set into dataframe
test_df = pd.read_csv('data/KDDTest+.csv')

# Reset column names for training set
train_df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                    'num_access_files', 'num_outbound_cmds', 'is_host_login',
                    'is_guest_login', 'count', 'srv_count', 'serror_rate',
                    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                    'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                    'dst_host_srv_rerror_rate', 'subclass']

# Reset column names for testing set
test_df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                   'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                   'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                   'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                   'num_access_files', 'num_outbound_cmds', 'is_host_login',
                   'is_guest_login', 'count', 'srv_count', 'serror_rate',
                   'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                   'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                   'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                   'dst_host_same_src_port_rate',
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                   'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                   'dst_host_srv_rerror_rate', 'subclass']

cols = ['protocol_type', 'service', 'flag']

# One-hot encoding
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each, 1)
    return df


# Applying one hot encoding to df's
train_df_1 = one_hot(train_df, cols)
test_df_1 = one_hot(test_df, cols)


# Function to min-max normalize
def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy()  # do not touch the original df
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# Dropping subclass column for training  and testing set
tmp = train_df_1.pop('subclass')
tmp1 = test_df_1.pop('subclass')

# Normalizing training set
train_df_2 = normalize(train_df_1, train_df_1.columns)
# train_df_2

# Normalizing testing set
test_df_2 = normalize(test_df_1, test_df_1.columns)
# test_df_2

# Fixing labels for training set
classlist_train = []
check1_train = (
    "apache2", "back", "land", "neptune", "mailbomb", "pod", "processtable", "smurf", "teardrop", "udpstorm", "worm")
check2_train = ("ipsweep", "mscan", "nmap", "portsweep", "saint", "satan")
check3_train = ("buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm")
check4_train = (
    "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf", "sendmail", "Snmpgetattack", "spy",
    "snmpguess", "warezclient", "warezmaster", "xlock", "xsnoop")

DoSCount_train = 0
ProbeCount_train = 0
U2RCount_train = 0
R2LCount_train = 0
NormalCount_train = 0

for item in tmp:
    if item in check1_train:
        classlist_train.append("DoS")
        DoSCount_train = DoSCount_train + 1
    elif item in check2_train:
        classlist_train.append("Probe")
        ProbeCount_train = ProbeCount_train + 1
    elif item in check3_train:
        classlist_train.append("U2R")
        U2RCount_train = U2RCount_train + 1
    elif item in check4_train:
        classlist_train.append("R2L")
        R2LCount_train = R2LCount_train + 1
    else:
        classlist_train.append("Normal")
        NormalCount_train = NormalCount_train + 1

# Fixing labels for testing set
classlist_test = []
check1_test = (
    "apache2", "back", "land", "neptune", "mailbomb", "pod", "processtable", "smurf", "teardrop", "udpstorm", "worm")
check2_test = ("ipsweep", "mscan", "nmap", "portsweep", "saint", "satan")
check3_test = ("buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm")
check4_test = (
    "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf", "sendmail", "Snmpgetattack", "spy",
    "snmpguess", "warezclient", "warezmaster", "xlock", "xsnoop")

DoSCount_test = 0
ProbeCount_test = 0
U2RCount_test = 0
R2LCount_test = 0
NormalCount_test = 0

for item in tmp1:
    if item in check1_test:
        classlist_test.append("DoS")
        DoSCount_test = DoSCount_test + 1
    elif item in check2_test:
        classlist_test.append("Probe")
        ProbeCount_test = ProbeCount_test + 1
    elif item in check3_test:
        classlist_test.append("U2R")
        U2RCount_test = U2RCount_test + 1
    elif item in check4_test:
        classlist_test.append("R2L")
        R2LCount_test = R2LCount_test + 1
    else:
        classlist_test.append("Normal")
        NormalCount_test = NormalCount_test + 1

print("Normal: ", NormalCount_test)
print("DoS: ", DoSCount_test)
print("Probe: ", ProbeCount_test)
print("U2R: ", U2RCount_test)
print("R2L: ", R2LCount_test)
print('\n')

# Appending class column to training set
train_df_2["Class"] = classlist_train
# train_df_2

# Appending class column to testing set
test_df_2["Class"] = classlist_test
# test_df_2

y_train = train_df_2['Class']
y_test = test_df_2['Class']

X_train = train_df_2.drop('Class', 1)
X_test = test_df_2.drop('Class', 1)

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
train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size=0.25, random_state=101)

from sklearn.preprocessing import MinMaxScaler

X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

x_columns_train = train_df_2.columns.drop('Class')
x_train_array = train_X[x_columns_train].values
x_train_1 = np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))

dummies = pd.get_dummies(train_y)  # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y_train_1 = dummies.values

print(x_train_1.shape, y_train_1.shape)

history = model_lstm.fit(x_train_1, y_train_1, epochs=20, batch_size=64)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

pred1 = model_lstm.predict(x_train_1)
pred2 = np.argmax(pred1, axis=1)
pred = le.fit_transform(pred2)
# pred = le.inverse_transform(pred)
y_eval = np.argmax(y_train_1, axis=1)
score = metrics.accuracy_score(y_eval, pred)
print("Validation score: {}%".format(score * 100))

acc = accuracy_score(y_eval, pred)
print("accuracy : ", acc)
recall = recall_score(y_eval, pred, average='weighted', zero_division=0)
print("recall : ", recall)
precision = precision_score(y_eval, pred, average='weighted', zero_division=0)
print("precision : ", precision)
f1_scr = f1_score(y_eval, pred, average='weighted', zero_division=0)
print("f1_score : ", f1_scr)

print("####   0:Dos  1:normal  2:Probe  3:R2L  4:U2L  ###\n\n")
print(classification_report(y_eval, pred, zero_division=0))

cm = confusion_matrix(y_eval, pred2)
print(cm, '\n')

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
