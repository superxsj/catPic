import skimage

from lr_utils import load_dataset
import matplotlib.pyplot as plt
import numpy as np

train_X_orig, train_y, test_X_orig, test_y, classes = load_dataset()

print('train_X'+str(train_X_orig.shape)+'\n')
print('train_y'+str(train_y.shape)+'\n')
print('test_x'+str(test_X_orig.shape)+'\n')
print('test_y'+str(test_y.shape)+'\n')

#数据处理
train_deal_X_orig = train_X_orig.reshape(train_X_orig.shape[0],-1).T
test_deal_X_orig = test_X_orig.reshape(test_X_orig.shape[0],-1).T
print('deal_X'+str(train_deal_X_orig.shape))
print('dea_x_test'+str(test_deal_X_orig.shape))

trian_set_X = 1.0*train_deal_X_orig/255
test_set_X = 1.0*test_deal_X_orig/255

# sigmoid 函数
def sigmoid(z):
    a = 1./(1+np.exp(-z))#np.exp 是 numpy 提供的求自然指数的函数
    return a


# Initialize parameters
def initialize_parameters(dim):
    w = np.zeros(shape=(dim, 1), dtype=np.float32)#np.zeros((a,b))，用于生成一个形状为 a×b 的零矩阵。
    b = 0
    return w, b

def propagate(w,b,X,y):
    m = X.shape[1]

    # 向前传播
    A = sigmoid(np.dot(w.T, X) + b)

    cost = (-1./m)*np.sum(y*np.log(A)+(1-y)*np.log(1-A),axis=1) # axis = 1 代表按行求和
    dw = (1./m)*np.dot(X, ((A-y).T))
    db = (1./m)*np.sum(A-y, axis = 1)

    cost = np.squeeze(cost)  # trans cost from matrix 1x1 into a real number

    return dw, db, cost


def optimized(w, b, X, Y, train_times, learning_rate):
    costs = []

    for i in range(train_times):

        dw, db, cost = propagate(w, b, X, Y)

        # 参数迭代
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本函数
        if i % 100 == 0:
            costs.append(cost)

    return w, b, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    # 确保矩阵维数匹配
    w = w.reshape(X.shape[0], 1)

    # 计算 Logistic 回归
    A = sigmoid(np.dot(w.T, X) + b)

    # 如果结果大于0.5，打上标签"1"，反之记录为"0"
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    # 确保一下矩阵维数正确
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def TestMyModel(X_train, Y_train, X_test, Y_test, train_times=100, learning_rate=0.005):
    # 初始化参数
    w, b = initialize_parameters(X_train.shape[0])

    # 开始梯度下降训练
    w, b, costs = optimized(w, b, X_train, Y_train, train_times, learning_rate)

    # 训练完成，测试一下效果
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # 输出效果
    print("Accuracy on train_set: " + str(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) + "%")
    print("Accuracy on test_set: " + str(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) + "%")
    print(Y_prediction_test)
    return w, b, costs, Y_prediction_test

def isCat(my_image = "my_image.jpg", dirname = "images/"):#判断自己照片

    # 初始化图片路径
    fname = dirname + my_image

    # 读取图片
    image = np.array(plt.imread(fname))

    # 将图片转换为训练集一样的尺寸
    num_px = trian_set_X.shape[2]
    my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T

    # 预测
    my_predicted_image = predict(w, b, my_image)

    # 绘图与结果输出
    plt.imshow(image)
    print("y = " + str(int(my_predicted_image)) + ", our neutral network predicts a \"" + classes[int(my_predicted_image),] +  "\" picture.")

if __name__ == '__main__':
    w, b, costs, Y_prediction_test = TestMyModel(trian_set_X, train_y, test_set_X, test_y, 2000, 0.002)
    plot_costs = np.squeeze(costs)
    plt.plot(plot_costs)
    plt.show()

#index = 102

#plt.imshow(train_X_orig[index])
#plt.show()