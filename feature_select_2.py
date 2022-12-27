import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Convolution1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
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

datasets = [train_df, test_df]
data_df = pd.concat(datasets, axis=0, ignore_index=True)

# xử lí row trùng
data_df.drop_duplicates(inplace=True, keep=False, ignore_index=True)
# print(data_df.shape)

# xử lý giá trị bị missing
data_df.dropna(axis=0, inplace=True, how="any")

# xử lý giá trị infinity
# Replace infinite values to NaN
data_df.replace([-np.inf, np.inf], np.nan, inplace=True)

# Remove infinte values
data_df.dropna(axis=0, inplace=True, how="any")


def remove_constant_features(data, threshold=0.01):
    # constant feature: chỉ có 1 giá trị cho tất cả các samples

    # Standard deviation denoted by sigma (σ) is the average of the squared root differences from the mean.
    data_std = data.std(numeric_only=True)

    # Find Features that meet the threshold
    constant_features = [column for column, std in data_std.iteritems() if std < threshold]
    print(constant_features)

    # Drop the constant features
    data.drop(labels=constant_features, axis=1, inplace=True)
    return data

import matplotlib.pyplot as plt
import seaborn as sns

def remove_correlated_features(data, threshold=0.98):
    # correlated freature: nếu giữa 2 hoặc nhiều feature có sự tương quan cao
    # có nghĩa là ta có thể suy ra cái còn lại từ 1 cái đã cho, nghĩa là feature thứ 2
    # không mạng lại thêm thông tin gì cho việc dự đoán target => bỏ cái thứ 2 đi

    # Correlation matrix
    data_corr = data.corr()
    fig = plt.figure(figsize=(15, 15))
    sns.set(font_scale=1.0)
    ax = sns.heatmap(data_corr, annot=False)
    fig.savefig('correlation_matrix.png')
    plt.close()

    # Create & Apply mask
    mask = np.triu(np.ones_like(data_corr, dtype=bool))
    tri_df = data_corr.mask(mask)

    # Find Features that meet the threshold
    correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
    print(correlated_features)

    # Drop the highly correlated features
    data.drop(labels=correlated_features, axis=1, inplace=True)
    return data

data_df = remove_constant_features(data_df)
data_df = remove_correlated_features(data_df)

tmp = [c for c in data_df.columns]
print(tmp)
print(len(tmp))


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



data_df = one_hot(data_df, cols)


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

atk_name = data_df.pop('subclass')

data_df = normalize(data_df, data_df.columns)

classlist_train = []
check1_train = (
    "apache2", "back", "land", "neptune", "mailbomb", "pod", "processtable", "smurf", "teardrop", "udpstorm", "worm")
check2_train = ("ipsweep", "mscan", "nmap", "portsweep", "saint", "satan")
check3_train = ("buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm")
check4_train = (
    "ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf", "sendmail", "Snmpgetattack", "spy",
    "snmpguess", "warezclient", "warezmaster", "xlock", "xsnoop")

for item in atk_name:
    if item in check1_train:
        classlist_train.append("DoS")
    elif item in check2_train:
        classlist_train.append("Probe")
    elif item in check3_train:
        classlist_train.append("U2R")
    elif item in check4_train:
        classlist_train.append("R2L")
    else:
        classlist_train.append("Normal")

data_df["Class"] =  classlist_train
Y = data_df['Class']
X = data_df.drop('Class', axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(Y)

'''
    0: DoS
    1: Normal
    2: Probe
    3: R2L
    4: U2R
'''

Y = pd.DataFrame(labelEncoder.transform(Y), columns=['Class'])

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

from sklearn.decomposition import PCA

pca = PCA(0.95)
pca.fit(X_train)
print("Cumulative Variances (Percentage):")
print(pca.explained_variance_ratio_.cumsum() * 100)

components = len(pca.explained_variance_ratio_)

# Số component sau khi dùng PCA
print(f'\nNumber of components: {components}')

plt.plot(range(1, components + 1),
         np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")
plt.savefig('pca_explained_variance.png')
plt.close()



pca_components = abs(pca.components_)
print('\nTop 4 most important features in each component')
print('===============================================')
for row in range(pca_components.shape[0]):
    # get the indices of the top 4 values in each row
    temp = np.argpartition(-(pca_components[row]), 4)

    # sort the indices in descending order
    indices = temp[np.argsort((-pca_components[row])[temp])][:4]

    # print the top 4 feature names
    print(f'Component {row + 1}: {X_train.columns[indices].to_list()}')


X_train_pca = pd.DataFrame(pca.transform(X_train))
X_test_pca = pd.DataFrame(pca.transform(X_test))

loadings = pd.DataFrame(
    data=pca.components_.T * np.sqrt(pca.explained_variance_),
    columns=[f'PC{i}' for i in range(1, len(X_train_pca.columns) + 1)],
    index=X_train.columns
)


pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
pc1_loadings = pc1_loadings.reset_index()
pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
plt.title('PCA loading scores (first principal component)', size=12)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig('pca.png')
plt.close()

# X_train_pca.to_pickle('X_train_pca.pkl')
# X_test_pca.to_pickle('X_test_pca.pkl')
# Y_train.to_pickle('Y_train.pkl')
# Y_test.to_pickle('Y_test.pkl')

# ====================

# def create_lstm_model(input_shape, num_classes):
#     model = Sequential()

#     model.add(Convolution1D(64, 3, padding="same", activation="relu", input_shape=input_shape))
#     model.add(Convolution1D(64, 3, padding="same", activation="relu"))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Convolution1D(128, 3, padding="same", activation="relu"))
#     model.add(Convolution1D(128, 3, padding="same", activation="relu"))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(LSTM(64, return_sequences=True))
#     model.add(Dropout(0.1))
#     model.add(LSTM(64, return_sequences=False))
#     model.add(Dropout(0.1))
#     model.add(Dense(48, activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(48, activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(num_classes, activation='softmax'))

#     return model


# input_shape = (122, 1)
# num_classes = 5
# model_lstm = create_lstm_model(input_shape, num_classes)
# # define optimizer and objective, compile lstm
# model_lstm.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# model_lstm.summary()

# # Split data: 75% training and 25% testing
# train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size=0.25, random_state=101)

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

# history = model_lstm.fit(x_train_1, y_train_1, epochs=20, batch_size=64)

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
