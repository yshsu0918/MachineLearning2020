import struct
import numpy as np
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

from collections import Counter
import time

def KNN(test,train_images,train_labels, K = 1):
    
    result = []
    for i in range(60000):
        sum = np.sum( (train_images[i] - test)*(train_images[i] - test) ) #歐幾里得距離
        result.append((sum,train_labels[i]))
        
    sorted_result = sorted(result,key=lambda t:t[0]) #根據distance 由小到大排序
    
    votes = [x[1] for x in sorted_result[0:K]] #取前K個最近的鄰居的label
    
    vote_counts = Counter(votes)
    top1 = vote_counts.most_common(1) #找出0-9哪個被投票最多次
    
    return top1[0][0]
        
    
def testKNN(K,test_size):
    correct = 0
    wrong = 0
    for i in range(test_size):
        predict_label= KNN(test_image[i], train_image, train_label,K=K)
        #print(predict_label,test_label[i])
        if predict_label == test_label[i]:
            correct += 1.0
        else:
            wrong += 1.0
    print('KNN', K, 'Accuracy', correct/(correct+wrong))

tStart = time.time()
testKNN(1,10000)
tEnd = time.time()
print('KNN1 use {} seconds'.format(tEnd - tStart))
tStart = time.time()
testKNN(5,10000)
tEnd = time.time()
print('KNN5 use {} seconds'.format(tEnd - tStart))