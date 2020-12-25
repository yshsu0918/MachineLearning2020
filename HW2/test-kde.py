from sklearn import svm
import numpy as np
import struct
from sklearn.neighbors.kde import KernelDensity
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

class_of_mnist = dict()

#把0-9分成九類
for i, x in enumerate( x_train ):
    if y_train[i] in class_of_mnist:
        class_of_mnist[y_train[i]].append(x)
    else:
        class_of_mnist[y_train[i]] = [ x ]
#列印出每一類有多少個 確定自己的程式對不對
for key in class_of_mnist.keys():
    print(key, len(class_of_mnist[key]))
#算出0-9 的分布機率

tStart = time.time()

kdes = []
for key in class_of_mnist.keys():
    kde = KernelDensity(kernel='gaussian', bandwidth = 0.2).fit(class_of_mnist[key])
    kdes.append( (kde,key) )

correct = 0
wrong = 0
for i, x in enumerate( x_test ):
    result = []
    for kde,key in kdes:
        score = kde.score_samples( [x] )
        result.append( (key, score) )
    #按照分數排序
    sorted_result = sorted(result, key=lambda x: x[1], reverse = True)
    #取最高分統計
    if sorted_result[0][0] == y_test[i]:
        correct += 1
    else:
        wrong += 1
        
print('KDE Accuracy', correct/(correct+wrong))
tEnd = time.time()
print('KDE use {} seconds'.format(tEnd - tStart))
