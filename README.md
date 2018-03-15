# svm
* svm.py <br>
该文件中实现了一个简单的SVM，使用SMO进行优化，在选择优化的变量时采用随机选择的方式。
* plattSMO.py <br>
该文件也是采用SMO进行优化，在选择优化变量时，选择误差步长最大的两个变量进行优化，可以大幅提高优化速度。
该文件中还加入了核函数（线性核函数，RBF核函数），具体实现参见 kernelTrans(self,x,z)
* libSVM.py <br>
该文件实现了一个SVM多分类器,其实现原理是：对于样本中的每两个类别之间都训练一个SVM二分类器。对于k个类别，
共可训练出k(k-1)/2个SVM二分类器。在预测时，将测试样例分别输入到k(k-1)/2分类器中。<br>
假设（i,j)表示划分类别i和类别j的SVM分类器<br>
对于每个分类器(i,j)：<br>
若分类结果为+1，则count[i] +=1<br>
若分类结果为-1，则count[j] +=1<br>
最后分类结果取相应类别计数最大的那个类别作为最终分类结果<br>
本文件还实现了将训练的模型保存成文件，方便预测时直接从文件读取，省去了再次训练的时间。<br>
** 例子
```python
def main():
    '''
    data,label = loadImage('trainingDigits')
    svm = LibSVM(data, label, 200, 0.0001, 10000, name='rbf', theta=20)
    svm.train()
    svm.save("svm.txt")
    '''
    svm = LibSVM.load("svm.txt")
    test,testlabel = loadImage('testDigits')
    svm.predict(test,testlabel)
```

