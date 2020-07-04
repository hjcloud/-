import operator
import struct
import numpy as np
import matplotlib.pyplot as plt

#读取图片
def read_image(filename):
    # 以二进制打开
    bin_data = open(filename, 'rb').read()
    offset = 0
    fmt_header = '>iiii' 

    #大端转化4个int变量
    #num_magic:魔数
    #num_images:图片数量
    #num_rows:图片行数
    #num_cols:图片列数
    num_magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    image_size = num_rows * num_cols  #784个像素点
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))

    #数组保存图片信息
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))

        #每张图片保存在28*28的数组中
        offset += struct.calcsize(fmt_image)
    print('读取图片完成')
    return images

 #读取标签
def read_label(filename):
    bin_data = open(filename, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    #大端转化2个int变量
    num_magic, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'

    #标签为1字节
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]

        #标签保存至一维数组中
        offset += struct.calcsize(fmt_image)
    print('读取标签完成')
    return labels

def knn(test_data,train_data,labels,k):
    trainsize = train_data.shape[0]
    diffMat = np.tile(test_data,(trainsize,1))-train_data
    sqDiffMat = diffMat**2
    #行相加
    sqDistances = sqDiffMat.sum(axis=1) 
    distances = sqDistances ** 0.5
    #距离从小到大排序
    sortedDistIndicies = distances.argsort()
    classCount = {}
    #前K个距离最小的标签存入字典，对应键值为标签出现次数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #存入字典并出现记录次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #按字典键值从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

if __name__ == '__main__':
    #处理数据
    train_images = read_image('train-images.idx3-ubyte')
    train_labels = read_label('train-labels.idx1-ubyte')
    test_images = read_image('test-images.idx3-ubyte')
    test_labels = read_label('test-labels.idx1-ubyte')

    print('开始识别')
    m = 60000  # 创建一个读入数据的数组，保存图片信息
    trainingMat = np.zeros((m, 784))  # 初始化为零

    #平展
    for i in range(m):
        for j in range(28):
            for k in range(28):
                trainingMat[i, 28*j+k] = train_images[i][j][k]
    mTest = 1000    

    #测试数量
    errorCount = 0.0

    #记录错误个数
    for i in range(mTest):
        classNumStr = test_labels[i]
        vectorUnderTest = np.zeros(784)

        #数组保存测试图片信息        
        for j in range(28):
            for k in range(28):
                vectorUnderTest[28*j+k] = test_images[i][j][k]  #存入第i个测试图

        #对测试图分类
        Result = knn(vectorUnderTest, trainingMat, train_labels, 3)
        if (Result != classNumStr):
            errorCount += 1.0

    print("\n错误数： %d" % errorCount)
    print("\n正确率率： %f" % ((mTest-errorCount) / float(mTest)))
    print ('处理结束')