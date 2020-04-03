import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, regularizers, Convolution1D, MaxPooling1D, Flatten
from keras.utils import to_categorical, plot_model
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

def re_pro():
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    import numpy as np
    import random as rn

    sd = 1  # Here sd means seed.
    np.random.seed(sd)
    rn.seed(sd)
    os.environ['PYTHONHASHSEED'] = str(sd)

    from keras import backend as K
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(sd)
    sess = tf.Session(graph=tf.get_default_graph(), config=config)
    K.set_session(sess)
re_pro()
def mat_plt(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()


def roc(fpr, tpr, roc_auc):
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.scatter(fpr, tpr, c='r', label='AUC2 = %0.2f' % roc_auc)
    # plt.plot(fpr2,tpr2,'r',label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend()
    plt.show()


def plot_confusion_matrix(confusion_m):
    plt.imshow(confusion_m, cmap=plt.cm.Blues)
    plt.colorbar()
    indices = range(len(confusion_m))
    classes = list(set(y_test))
    labels_x = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconna', 'Analysis', 'Backdoor', 'Shellcode',
                'Worms']
    labels_y = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 'Analysis', 'Backdoor',
                'Shellcode', 'Worms']
    # plt.xticks(indices, classes)
    # plt.yticks(indices, classes)
    plt.xticks(indices,labels_x,rotation=45)
    plt.yticks(classes,labels_y)
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.autolayout'] = True
    for i in range(len(confusion_m)):
        for j in range(len(confusion_m[i])):
            plt.text(j, i, confusion_m[i][j])
    plt.show()

data = pd.read_csv('../paper/all.csv')
data2 = data.iloc[:, 1:-2]
data_ytwo = data.iloc[:, -1]
data_yfive = data.iloc[:, -2]
# data_x_xg = data_x['ct_dst_sport_ltm','tcprtt','dwin','ct_src_dport_ltm',
# 'ct_dst_src_ltm','ct_dst_ltm','smean','dmean','dtcpb']
columnsxx = ['sbytes','dur','smean','ct_srv_src','ct_src_ltm',
             'sload','ct_dst_src_ltm','dmean','ct_dst_ltm',
             'dbytes','response_body_len','stcpb',
             'ct_srv_dst','synack','dload','rate','dtcpb',
             'sinpkt','sjit','ct_src_dport_ltm','proto','service','state']
columnsx12 = ['ct_dst_sport_ltm','tcprtt','dwin','ct_src_dport_ltm',
'ct_dst_src_ltm','ct_dst_ltm','smean','dmean','dtcpb','proto','service','state']
columns9 = ['ct_dst_sport_ltm','tcprtt','dwin','ct_src_dport_ltm',
'ct_dst_src_ltm','ct_dst_ltm','smean','dmean','dtcpb']
column_16 = ['sbytes','dtcpb','ackdat','dur','synack','sload',
'smean','tcprtt','ct_srv_dst','sjit','dinpkt',
'rate','sinpkt','ct_dst_ltm','stcpb','proto']
column_q = ['ct_dst_sport_ltm','tcprtt','dwin','ct_src_dport_ltm',
'ct_dst_src_ltm','ct_dst_ltm','smean','dmean','dtcpb',]
# data_x_xgboost = pd.DataFrame(data2,columns=column_q)
# data_x = pd.get_dummies(data_x_xgboost)
data_x = pd.get_dummies(data2)
data_two = pd.concat([data_x, data_ytwo], axis=1)
data_five = pd.concat([data_x, data_yfive], axis=1)
# scaler_2 = MinMaxScaler(feature_range=(0, 1))  #自动将dtype转换成float64
# data_two = scaler_2.fit_transform(data_two)
# index_train = np.arange(175341)
# np.random.shuffle(index_train)
data_train = np.array(data_five.iloc[:175341, :])
from sklearn.model_selection import ShuffleSplit
rs = ShuffleSplit(n_splits=1, test_size=0.3, random_state=1)

for train_1,train_2 in rs.split(data_train):
    train_70 = data_train[train_1,:]
    train_30 = data_train[train_2,:]
train_70_x = train_70[:,:-1]
train_70_y = train_70[:,-1]
train_30_x = train_30[:,:-1]
train_30_y = train_30[:,-1]
x_test = np.array(data_five.iloc[175341:, :-1])
y_test = np.array(data_five.iloc[175341:, -1])
# x_train = x_train[index_train]
# y_train = y_train[index_train]

scaler_2 = MinMaxScaler(feature_range=(0, 1))  #自动将dtype转换成float64
train_70_x = scaler_2.fit_transform(train_70_x)
train_30_x = scaler_2.transform(train_30_x)
x_test = scaler_2.transform(x_test)
train_70_x = np.reshape(train_70_x, (train_70_x.shape[0],train_70_x.shape[1],1))
train_30_x = np.reshape(train_30_x, (train_30_x.shape[0],train_30_x.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
train_70_y = to_categorical(train_70_y,10)
train_30_y = to_categorical(train_30_y,10)
y_test2 = to_categorical(y_test,10)
model = Sequential()
model.add(Convolution1D(32, 3, border_mode="same",activation="relu",input_shape=(196, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.15))
model.add(Convolution1D(32, 3, border_mode="same", activation="relu"))
model.add(MaxPooling1D(pool_length=(2)))
model.add(BatchNormalization())
model.add(Dropout(0.14))
model.add(Convolution1D(64, 3, border_mode="same", activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.14))
model.add(Convolution1D(64, 3, border_mode="same", activation="relu"))
model.add(MaxPooling1D(pool_length=(2)))
model.add(BatchNormalization())
model.add(Dropout(0.14))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.14))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', metrics=['acc'], loss='categorical_crossentropy')
model.summary()
plot_model(model, to_file='model3.png', show_layer_names=True, show_shapes=True)
def create_class_weight(labels_dict, mu=0.1):
    total = np.sum([i for i in labels_dict.values()])
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = math.log((mu * total) / float(labels_dict[key]))
        math.log(mu)
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight
labels_dict = {0: 37000, 1: 18871, 2: 11132, 3: 6062, 4: 4089, 5: 3496, 6: 677, 7: 583
    , 8: 378, 9: 44}
labels_dict_train ={0: 56000, 1: 40000, 2: 33393, 3: 18184, 4: 12264, 5: 10491, 6: 2000, 7: 1746
                   ,8:1133,9:130}
class_weight_dict = create_class_weight(labels_dict_train)
history = model.fit(train_70_x, train_70_y,validation_data=(train_30_x,train_30_y),batch_size=4096, epochs=50)
model.save('my_model2.h5')
loss, accuracy = model.evaluate(x_test, y_test2)
pre_y = model.predict_classes(x_test)
y_test = np.array(y_test)
metrics = classification_report(y_test, pre_y)
print(metrics)
confusion_m = confusion_matrix(y_test, pre_y)
print(confusion_m)
y_pred_pro = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_pro, pos_label=1)
roc_auc = auc(fpr, tpr)
mat_plt(history)
plot_confusion_matrix(confusion_m)
roc(fpr, tpr, roc_auc)
model.save('my_model2.h5')
def fpr_tpr(confusion_m):
    sum = 0
    count = 0
    k = 0
    lubao = []
    for i in confusion_m:
        for j in i:
            sum = sum + j
        acc = float(i[count]) / float(sum)
        if k == 0:
            wubao = float(sum - i[0]) / float(sum)
            print('误报率', wubao)
        if k > 0:
            loubao = float(i[0]) / float(sum)
            print('漏报率', loubao)
        count = count + 1
        k = k + 1
        sum = 0
        print('各个准确率%f' % acc)

fpr_tpr(confusion_m)
print('Loss:%f' % loss, 'Accuracy:%f' % accuracy)
print('第%d个epoch准确率最高' % history.history['val_acc'].index(max(i for i in history.history['val_acc'])),
      max(i for i in history.history['val_acc']))