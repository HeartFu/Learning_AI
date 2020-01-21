import numpy as np
import matplotlib.pyplot as plt
import os, re
import cv2
import random
import tensorflow as tf
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
IMG_SIZE = 28
class NeuralNetwork:
    def __init__(self):

        self.model = Sequential()
        self.model.add(Dense(IMG_SIZE * IMG_SIZE))
        self.model.add(Dense(500, activation='relu'))
        self.model.add(Dropout(0.7))
        self.model.add(Dense(7, activation='softmax'))
        sgd = optimizers.SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=False)
        self.model.compile(optimizer=sgd,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    def train(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train,
                       batch_size=100,
                       verbose=1,
                       epochs=500, validation_data=(x_test, y_test))

    def predict(self, x_test):
        predTest = self.model.predict(x_test)
        return predTest

def sort_key(s):
    #sort_strings_with_embedded_numbers
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)  # 切成数字与非数字
    pieces[1::2] = map(int, pieces[1::2])  # 将数字部分转成整数
    # return pieces[len(pieces) - 2]
    return pieces


def load_ck_data(path):
    data_path = 'cohn-kanade-images'
    labels_path = 'Emotion'
    data_path_comp = os.path.join(path, data_path)
    labels_path_comp = os.path.join(path, labels_path)
    img_list = []
    for dir in os.listdir(data_path_comp):
        if dir == '.DS_Store':
            continue
        if os.path.splitext(dir)[-1] != '.png':
            data_path_person = os.path.join(data_path_comp, dir)
            for dir_1 in os.listdir(data_path_person):
                if dir_1 == '.DS_Store':
                    continue
                if os.path.splitext(dir_1)[-1] != '.png':
                    data_path_person_img = os.path.join(data_path_person, dir_1)
                    # get the path and insteal the Label dir
                    label_path_person_img = data_path_person_img.replace(data_path, labels_path)
                    label_target = ''
                    try:
                        flag = 0
                        for label in os.listdir(label_path_person_img):
                            label_file = open(os.path.join(label_path_person_img,label), "r")
                            lines = label_file.readlines()
                            for line in lines:
                                label_target += line.strip()
                            flag = 1
                    except BaseException:
                        continue
                    if flag == 1:
                        # already have label
                        list = os.listdir(data_path_person_img)
                        if len(list) >= 10:
                            range_i = len(list) - 3
                        else:
                            range_i = len(list) - 1
                        list.sort(key=sort_key)
                        for i in range(range_i, len(list)):
                            img = list[i]
                            if img == '.DS_Store':
                                continue
                            if os.path.splitext(img)[-1] != '.png':
                                continue
                            img_array = cv2.imread(os.path.join(data_path_person_img, img), cv2.IMREAD_GRAYSCALE)
                            new_array = cv2.resize(img_array, (28, 28))
                            img_final = np.ndarray.flatten(np.asarray(new_array, dtype='float64') / 255)  # 将图像转化为数组并将像素转化到0-1之间
                            img_list.append([img_final, label_target])
                            # augmentation flip horizontally
                            img_flip_hor = cv2.flip(new_array, 1)
                            img_flip_hor_final = np.ndarray.flatten(
                                np.asarray(img_flip_hor, dtype='float64') / 255)  # 将图像转化为数组并将像素转化到0-1之间
                            img_list.append([img_flip_hor_final, label_target])
                            # augmentation transpose
                            img_transpose = cv2.transpose(new_array)
                            img_transpose_final = np.ndarray.flatten(
                                np.asarray(img_transpose, dtype='float64') / 255)  # 将图像转化为数组并将像素转化到0-1之间
                            img_list.append([img_transpose_final, label_target])

    random.shuffle(img_list)
    images_ori = []
    labels = []
    for img_np, label in img_list:
        images_ori.append(img_np)
        labels.append(label)
    print('dataset length is {}'.format(len(labels)))
    return np.array(images_ori).astype(np.float64), np.array(labels).astype(np.float64).astype(int)

# evaluation
def eval_model(y_true, y_pred, classes):
    # caculate the Precision, Recall, f1, support for every class
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': classes,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': [u'总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [999]
    res = pd.concat([res1, res2])
    # caculate the comfusion matrix
    # conf_mat = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=labels, index=labels)
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm_normalized, res[[u'Label', u'Precision', u'Recall', u'F1', u'Support']]


def plot_confusion_matrix(cm, classes, save_dir, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.figure()
    tick_marks = np.array(range(len(classes))) + 0.5
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)

    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=45)
    plt.yticks(xlocations, classes)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=12, horizontalalignment="center", verticalalignment="center")
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_dir, format='png')
    plt.close()

if __name__ == "__main__":
    # the path of datasets, if you want to run, you need to modify to your datasets path
    path_ck_unpack = 'datasets/CK+'
    classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    # load data
    images, labels = load_ck_data(path_ck_unpack)
    labels = labels - 1

    # change label [5] to [0,0,0,0,1,0,0]
    one_hots = to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, one_hots, test_size=0.2, random_state=20)

    nn = NeuralNetwork()
    nn.train(x_train, y_train, x_test, y_test)
    nn.model.summary()

    train_pre = nn.predict(x_train)
    data_train_pre = [np.argmax(one_hot) for one_hot in train_pre]
    data_train = [np.argmax(one_hot) for one_hot in y_train]
    accuracy = accuracy_score(data_train_pre, data_train)
    conf_mat, evalues = eval_model(data_train, data_train_pre, classes)
    plt.figure()
    plot_confusion_matrix(conf_mat, classes, 'train_cm.png', title='CM on Train Set')
    print('train confusion matrix is: {}'.format(conf_mat))
    print('train evalues is :{}'.format(evalues))
    print('Train accuracy is : {}'.format(accuracy))

    test_pre = nn.predict(x_test)
    data_test_pre = [np.argmax(one_hot) for one_hot in test_pre]
    data_test = [np.argmax(one_hot) for one_hot in y_test]
    accuracy = accuracy_score(data_test_pre, data_test)
    conf_mat, evalues = eval_model(data_test, data_test_pre, classes)
    plot_confusion_matrix(conf_mat, classes, 'test_cm.png', title='CM on Test Set')
    print('train confusion matrix is: {}'.format(conf_mat))
    print('train evalues is :{}'.format(evalues))

    print('Train accuracy is : {}'.format(accuracy))






