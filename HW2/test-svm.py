from sklearn import svm
import numpy as np
import struct
import time
def decode_idx3_ubyte(idx3_ubyte_file,dataset_size):
    f = open(idx3_ubyte_file, 'rb').read()
    mem_offset = 16
    images = []
    for i in range(dataset_size):
        if (i+1) % (dataset_size/100) == 0:
            print('#', end='')
        images.append( np.array(struct.unpack_from('>784B', f, mem_offset)).reshape((28, 28)))
        mem_offset += (784)
    return images
    
def decode_idx1_ubyte(idx1_ubyte_file,dataset_size):
    f = open(idx1_ubyte_file, 'rb').read()
    mem_offset = 8
    labels = []
    for i in range(dataset_size):
        if (i+1) % (dataset_size/100) == 0:
            print('#', end='')
        labels.append( struct.unpack_from('>B', f, mem_offset)[0] )
        mem_offset += 1
    return labels


train_image = decode_idx3_ubyte('train-images.idx3-ubyte',60000)
train_label = decode_idx1_ubyte('train-labels.idx1-ubyte',60000)
print('load train done')
test_image = decode_idx3_ubyte('t10k-images.idx3-ubyte',10000)
test_label = decode_idx1_ubyte('t10k-labels.idx1-ubyte',10000)
print('load test done')


#mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
train_num = 60000
test_num = 10000

x_train = [ x.reshape(28*28) for x in train_image]
y_train = train_label
x_test = [ x.reshape(28*28) for x in test_image]
y_test = test_label


tStart = time.time()
# 獲取一個支援向量機模型
print('1')
predictor = svm.SVC(kernel='linear', verbose=True, max_iter = 1000)
# 把資料丟進去
print('2')
predictor.fit(x_train[:train_num], y_train[:train_num])
# 預測結果
print('3')
result = predictor.predict(x_test[:test_num])
# 準確率估計
print('4')
accurancy = np.sum(np.equal(result, y_test[:test_num])) / test_num
print(accurancy)
tEnd = time.time()
print('SVM use {} seconds'.format(tEnd - tStart))
