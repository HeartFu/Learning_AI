import os, re
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


from sklearn.model_selection import train_test_split

INIT_W = 0.01 # 权值初始化

# 设置缺省数值类型
DTYPE_DEFAULT = np.float32

def softmax(y):
    # minus the max value to prevent the over exp
    max_y = np.max(y,axis=1)
    max_y.shape=(-1,1)
    y1 = y - max_y
    # compute the exp
    exp_y = np.exp(y1)
    sigma_y = np.sum(exp_y,axis = 1)
    sigma_y.shape=(-1,1)
    softmax_y = exp_y/sigma_y

    return softmax_y

softmax_activation_function = softmax

def sort_key(s):
    #sort policy
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)  # 切成数字与非数字
    pieces[1::2] = map(int, pieces[1::2])  # 将数字部分转成整数
    return pieces

def load_ck_data(path):
    data_path = 'cohn-kanade-images'
    labels_path = 'Emotion'
    data_path_comp = os.path.join(path, data_path)
    # labels_path_comp = os.path.join(path, labels_path)
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
                        # list.sort(key = lambda x: int(x[9:-4]))
                        list.sort(key=sort_key)
                        # print("list = {}".format(list))
                        for i in range(range_i, len(list)):
                            img = list[i]
                            # print(img)
                            if img == '.DS_Store':
                                continue
                            if os.path.splitext(img)[-1] != '.png':
                                continue
                            img_array = cv2.imread(os.path.join(data_path_person_img, img), cv2.IMREAD_GRAYSCALE)
                            new_array = cv2.resize(img_array, (28, 28))
                            img_final = np.ndarray.flatten(np.asarray(new_array, dtype='float64') / 255)  # 将图像转化为数组并将像素转化到0-1之间
                            img_list.append([img_final, label_target])
    random.shuffle(img_list)
    images_ori = []
    labels = []
    for img_np, label in img_list:
        images_ori.append(img_np)
        labels.append(label)
    print('dataset length is {}'.format(len(labels)))
    return np.array(images_ori).astype(np.float64), np.array(labels).astype(np.float64).astype(int)


# Loss curve
def showCurves(idx, x, ys, line_labels, colors,ax_labels):
    LINEWIDTH = 1.0
    plt.figure(figsize=(6, 6))
    #loss
    ax1 = plt.subplot(211)
    for i in range(2):
        line = plt.plot(x[:idx], ys[i][:idx])[0]
        plt.setp(line, color=colors[i],linewidth=LINEWIDTH, label=line_labels[i])

    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.set_xlabel(ax_labels[0])
    ax1.set_ylabel(ax_labels[1])
    plt.grid()
    plt.legend()

    #Acc
    ax2 = plt.subplot(212)
    for i in range(2,4):
        line = plt.plot(x[:idx], ys[i][:idx])[0]
        plt.setp(line, color=colors[i],linewidth=LINEWIDTH, label=line_labels[i])

    ax2.xaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.set_xlabel(ax_labels[0])
    ax2.set_ylabel(ax_labels[2])

    plt.grid()
    plt.legend()
    plt.show()

class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 init_learning_rate,
                 min_batch_size,
                 iteration,
                 momentum,
                 LOSS_CURVE_FLAG,
                 flag_number_train,
                 epoch_num,
                 active_percentage=0.0,
                 lamda=0.0,
                 lr_policy=False,
                 bias=None,
                 ):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes

        self.no_of_hidden_nodes = no_of_hidden_nodes

        self.init_learning_rate = init_learning_rate
        self.bias = bias
        self.min_batch_size = min_batch_size
        self.iteration = iteration # the number of train every batch
        self.momentum = momentum
        self.flag_number_train = flag_number_train

        self.create_weight_matrices()
        self.loss_curve_flag = LOSS_CURVE_FLAG
        self.epoch_num = epoch_num
        self.active_percentage = active_percentage
        self.lamda = lamda
        self.lr_policy = lr_policy
        if True == LOSS_CURVE_FLAG:
            self.cur_p_idx = 0
            self.curv_x = np.zeros(epoch_num * 100, dtype=int)
            self.curv_ys = np.zeros((4, epoch_num * 100), dtype=DTYPE_DEFAULT)

    def create_weight_matrices(self):
        # init weight, w：D*H matrix
        w = INIT_W * np.random.randn(self.no_of_in_nodes, self.no_of_hidden_nodes)  # D*K

        # w2 H*K matrix
        w2 = INIT_W * np.random.randn(self.no_of_hidden_nodes, self.no_of_out_nodes)
        self.w = w
        self.w2 = w2
        print('w,b inited..')

    def forward(self, x):
        # Relu function
        hidden_layer = np.maximum(0, np.matmul(x, self.w))
        # dropout forward
        if self.active_percentage != 0.0:
            mask = (np.random.rand(*hidden_layer.shape) > (1 - self.active_percentage)).astype(np.float32) / self.active_percentage
            self.mask = mask
            hidden_layer *= mask
        y = np.matmul(hidden_layer, self.w2)
        softmax_y = softmax(y)

        return hidden_layer, softmax_y

    def backward(self, softmax_y, curr_batch_size, values, hidden_layer, x):
        # print("------------------softmax_y-----------")
        # print(softmax_y)
        softmax_y[range(curr_batch_size), values] -= 1
        delta_y_mean = softmax_y / curr_batch_size

        delta_w2 = np.dot(hidden_layer.T, delta_y_mean)
        # not frist, need add momentum 正则化部分梯度
        if self.lamda != 0.0:
            delta_w2 = self.l2_regularizer_weight(self.w2, delta_w2)
            # delta_b2 = np.sum(delta_y_mean, axis=0, keepdims=True)
        if self.flag_number_train != 0:
            # not frist, need add momentum
            self.w2 = self.w2 - self.init_learning_rate * delta_w2 - self.momentum * self.last_delta_w2
            # self.b2 = self.b2 - self.init_learning_rate * delta_b2
        else:
            self.w2 = self.w2 - self.init_learning_rate * delta_w2
            # self.b2 = self.b2 - self.init_learning_rate * delta_b2

        self.last_delta_w2 = delta_w2

        dhidden = np.dot(delta_y_mean, self.w2.T)

        # dropout backward pass
        if self.active_percentage != 0.0:
            dhidden = dhidden * self.mask

        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        delta_w = np.dot(x.T, dhidden)
        # l2 正则化部分梯度
        if self.lamda != 0.0:
            delta_w += self.l2_regularizer_weight(self.w, delta_w)
        if self.flag_number_train != 0:
            # not frist, need add momentum
            self.w = self.w - self.init_learning_rate * delta_w - self.momentum * self.last_delta_w
        else:
            self.w = self.w - self.init_learning_rate * delta_w
            self.flag_number_train = 1
        self.last_delta_w = delta_w
        # print("-------------w2--------------------------")
        # print(self.w2)
        # print("-------------w---------------------------")
        # print(self.w)

    # learning rate decline on every epoch
    def lr_policy_fun(self, epoch):
        if True == self.lr_policy:
            if epoch != 0:
                self.init_learning_rate = self.init_learning_rate * 0.99

    # L2 regularizer loss
    def l2_regularizer_loss(self):
        reg_loss = self.lamda * (np.sum(self.w * self.w) + np.sum(self.w2 * self.w2))
        return reg_loss

    # L2 regularizer weight
    def l2_regularizer_weight(self, weight, delta_w):
        l2_weight = delta_w + self.lamda * weight
        return l2_weight


    def train_single(self, images, labels, epoch, sample_range,
                     batches_per_epoch, n_class,
                     images_v, labels_v):
        # learning rate declay 学习率指数衰减
        self.lr_policy_fun(epoch)

        rest_range = sample_range
        loss = 0.0
        accuracy_tr = 0.0
        loss_v = 0.0
        accuracy = 0.0
        for batch in range(batches_per_epoch):
            curr_batch_size = min(self.min_batch_size, len(rest_range))
            samples = random.sample(rest_range, curr_batch_size)
            rest_range = list(set(rest_range).difference(set(samples)))
            #   输入 N*D
            x = np.array([images[sample] for sample in samples], dtype=DTYPE_DEFAULT)
            #   正确类别 1*K
            values = np.array([labels[sample] for sample in samples])


            # 每个mini-batch进行I轮训练
            for i in range(self.iteration):

                hidden_layer, softmax_y = self.forward(x)

                # validate the every epoch
                if (batches_per_epoch - 1 == batch) and (self.iteration - 1 == i):
                    # train_loss
                    corect_logprobs = -np.log(softmax_y[range(curr_batch_size), values])
                    data_loss = np.sum(corect_logprobs) / curr_batch_size
                    loss = data_loss
                    # L2 regularizer
                    if self.lamda != 0.0:
                        reg_loss = self.l2_regularizer_loss()
                        loss += reg_loss

                    # train_acc
                    labels_pre_tr = np.argmax(softmax_y, axis=1)
                    accuracy_tr = np.mean(labels_pre_tr == values)

                    # accuracy of test
                    y_v_hidden_layer = np.maximum(0, np.matmul(images_v, self.w))
                    y_v = np.matmul(y_v_hidden_layer, self.w2)
                    # predict
                    labels_pre = np.argmax(softmax(y_v), axis=1)
                    accuracy = np.mean(labels_pre == labels_v)
                    # val loss
                    softmax_y_v = softmax(y_v)
                    corect_logprobs_v = -np.log(softmax_y_v[range(len(labels_v)), labels_v])
                    data_loss_v = np.sum(corect_logprobs_v) / len(labels_v)
                    loss_v = data_loss_v
                    if self.lamda != 0.0:
                        reg_loss = self.l2_regularizer_loss()
                        loss_v += reg_loss


                    if True == self.loss_curve_flag:

                        # curv_data_x
                        self.curv_x[self.cur_p_idx] = epoch
                        # train_loss
                        self.curv_ys[0][self.cur_p_idx] = loss
                        # val_loss
                        self.curv_ys[1][self.cur_p_idx] = loss_v
                        # train_acc
                        self.curv_ys[2][self.cur_p_idx] = accuracy_tr
                        # val_acc
                        self.curv_ys[3][self.cur_p_idx] = accuracy

                        self.cur_p_idx += 1

                    print('epoch %d , train_loss=%s, train_accuracy = %s, test_loss=%s, test_accuracy = %s' % (epoch, loss, accuracy_tr, loss_v, accuracy))

                # backward pass
                self.backward(softmax_y,curr_batch_size, values, hidden_layer, x)
        return loss, accuracy_tr, loss_v, accuracy,
    def train(self, train, labels, test, labels_v):
        print('start train')
        # train
        # sample calss
        n_class = self.no_of_out_nodes
        # sample range
        sample_range = [i for i in range(len(labels))]

        batches_per_epoch = int(np.ceil(len(labels) / self.min_batch_size))
        for epoch in range(self.epoch_num):
            loss_train, accuracy_train, loss_test, accuracy_test = self.train_single(train, labels, epoch, sample_range,
                         batches_per_epoch, n_class, test, labels_v)
        # show the loss and accuracy
        if True == self.loss_curve_flag:
            showCurves(self.cur_p_idx, self.curv_x, self.curv_ys, ['train_loss', 'val_loss', 'train_acc', 'val_acc'],
                       ['y', 'r', 'g', 'b'], ['Iteration', 'Loss', 'Accuracy'])

        return loss_train, accuracy_train, loss_test, accuracy_test

if __name__ == "__main__":
    print("start!")

    # the path of datasets, if you want to run, you need to modify to your datasets path
    path_ck_unpack = 'datasets/CK+'

    # load data
    images, labels = load_ck_data(path_ck_unpack)
    labels = labels - 1
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=20)

    # step 3. initialising the learning rate and momentum
    # ANN = NeuralNetwork(no_of_in_nodes=len(images[0]),
    #                     no_of_out_nodes=7,
    #                     no_of_hidden_nodes=500,
    #                     init_learning_rate=0.01,
    #                     min_batch_size=100,
    #                     iteration=10,
    #                     momentum=0.1,
    #                     flag_number_train=0,
    #                     LOSS_CURVE_FLAG=True,
    #                     epoch_num=50)
    # #
    # ANN.train(x_train, y_train, x_test, y_test)
    #
    # # step 4. find the best learning rate
    # init_learning_rate = 0.001
    # lr_list = []
    # curv_x = np.zeros(10, dtype=DTYPE_DEFAULT)
    # curv_ys = np.zeros((4, 10), dtype=DTYPE_DEFAULT)
    # for i in range (0, 10):
    #     learning_rate = init_learning_rate + 0.001 * i
    #     accuracy_tr = 0
    #     accuracy_te = 0
    #     loss_tr = 0
    #     loss_te = 0
    #     count = 10.0
    #     for j in range (0, 10):
    #         ANN = NeuralNetwork(no_of_in_nodes=len(images[0]),
    #                             no_of_out_nodes=7,
    #                             no_of_hidden_nodes=500,
    #                             init_learning_rate=learning_rate,
    #                             min_batch_size=100,
    #                             iteration=10,
    #                             momentum=0.1,
    #                             flag_number_train=0,
    #                             LOSS_CURVE_FLAG=False,
    #                             epoch_num=50)
    #
    #         loss_train, accuracy_train, loss_test, accuracy_test = ANN.train(x_train, y_train, x_test, y_test)
    #         accuracy_tr += accuracy_train
    #         accuracy_te += accuracy_test
    #         loss_tr += loss_train
    #         loss_te += loss_test
    #
    #     print('-----------------------------------------------------------------------------')
    #     print('final train_loss=%s, train_accuracy = %s, test_loss=%s, test_accuracy = %s' % (
    #         loss_tr / count, accuracy_tr / count, loss_te / count, accuracy_te / count))
    #     print('-----------------------------------------------------------------------------')
    #     curv_x[i] = learning_rate
    #     # train_loss
    #     curv_ys[0][i] = loss_tr / count
    #     # val_loss
    #     curv_ys[1][i] = loss_te / count
    #     # train_acc
    #     curv_ys[2][i] = accuracy_tr / count
    #     # val_acc
    #     curv_ys[3][i] = accuracy_te / count
    #
    # showCurves(10, curv_x, curv_ys, ['train_loss', 'val_loss', 'train_acc', 'val_acc'],
    #            ['y', 'r', 'g', 'b'], ['learning rate', 'Loss', 'Accuracy'])

    # step 4-1 learning rate policy
    # ANN = NeuralNetwork(no_of_in_nodes=len(images[0]),
    #                     no_of_out_nodes=7,
    #                     no_of_hidden_nodes=500,
    #                     init_learning_rate=0.01,
    #                     min_batch_size=100,
    #                     iteration=10,
    #                     momentum=0.1,
    #                     flag_number_train=0,
    #                     LOSS_CURVE_FLAG=True,
    #                     lr_policy=True,
    #                     epoch_num=50)
    # #
    # ANN.train(x_train, y_train, x_test, y_test)

    # # step 5 dropout
    # ANN = NeuralNetwork(no_of_in_nodes=len(images[0]),
    #                     no_of_out_nodes=7,
    #                     no_of_hidden_nodes=500,
    #                     init_learning_rate=0.01,
    #                     min_batch_size=100,
    #                     iteration=10,
    #                     momentum=0.1,
    #                     flag_number_train=0,
    #                     LOSS_CURVE_FLAG=True,
    #                     lr_policy = True,
    #                     active_percentage=0.7,
    #                     lamda=0,
    #                     epoch_num=50)
    # #
    # ANN.train(x_train, y_train, x_test, y_test)

    # step 5 L2
    # ANN = NeuralNetwork(no_of_in_nodes=len(images[0]),
    #                     no_of_out_nodes=7,
    #                     no_of_hidden_nodes=500,
    #                     init_learning_rate=0.01,
    #                     min_batch_size=100,
    #                     iteration=10,
    #                     momentum=0.1,
    #                     flag_number_train=0,
    #                     lr_policy=True,
    #                     LOSS_CURVE_FLAG=True,
    #                     lamda=0.0001,
    #                     epoch_num=50)
    # #
    # ANN.train(x_train, y_train, x_test, y_test)

    # step 6. Optimizing the topology of the network
    # 500 hidden node
    ANN = NeuralNetwork(no_of_in_nodes=len(images[0]),
                        no_of_out_nodes=7,
                        no_of_hidden_nodes=500,
                        init_learning_rate=0.01,
                        min_batch_size=100,
                        iteration=10,
                        momentum=0.1,
                        flag_number_train=0,
                        lr_policy=True,
                        LOSS_CURVE_FLAG=True,
                        active_percentage=0.7,
                        epoch_num=50)
    # 256 hidden nodes
    # ANN = NeuralNetwork(no_of_in_nodes=len(images[0]),
    #                     no_of_out_nodes=7,
    #                     no_of_hidden_nodes=500,
    #                     init_learning_rate=0.01,
    #                     min_batch_size=100,
    #                     iteration=10,
    #                     momentum=0.1,
    #                     flag_number_train=0,
    #                     lr_policy=True,
    #                     LOSS_CURVE_FLAG=True,
    #                     active_percentage=0.7,
    #                     epoch_num=50)
    # 512 hidden nodes
    # ANN = NeuralNetwork(no_of_in_nodes=len(images[0]),
    #                     no_of_out_nodes=7,
    #                     no_of_hidden_nodes=500,
    #                     init_learning_rate=0.01,
    #                     min_batch_size=100,
    #                     iteration=10,
    #                     momentum=0.1,
    #                     flag_number_train=0,
    #                     lr_policy=True,
    #                     LOSS_CURVE_FLAG=True,
    #                     active_percentage=0.7,
    #                     epoch_num=50)
    # 1024 hidden nodes
    # ANN = NeuralNetwork(no_of_in_nodes=len(images[0]),
    #                     no_of_out_nodes=7,
    #                     no_of_hidden_nodes=500,
    #                     init_learning_rate=0.01,
    #                     min_batch_size=100,
    #                     iteration=10,
    #                     momentum=0.1,
    #                     flag_number_train=0,
    #                     lr_policy=True,
    #                     LOSS_CURVE_FLAG=True,
    #                     active_percentage=0.7,
    #                     epoch_num=50)
    #
    ANN.train(x_train, y_train, x_test, y_test)


