import sys
from numpy import *
#from math import *
def loadDataSet(filename):
    fr = open(filename)
    data = []
    label = []
    for line in fr.readlines():
        lineAttr = line.strip().split('\t')
        data.append([float(x) for x in lineAttr[:-1]])
        label.append(float(lineAttr[-1]))
    return data,label

def selectJrand(i,m):
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j

def clipAlpha(a_j,H,L):
    if a_j > H:
        a_j = H
    if L > a_j:
        a_j = L
    return a_j

def smoSimple(data,label,C,toler,maxIter):
    dataMatrix = mat(data)
    labelMatrix = mat(label).transpose()
    b = 0.0
    iter = 0
    m,n = shape(dataMatrix)
    alpha = mat(zeros((m,1)))
    while iter < maxIter:
        alphapairChanged = 0
        for i in range(m):
            fxi = float(multiply(alpha,labelMatrix).T * (dataMatrix * dataMatrix[i,:].T)) + b
            Ei = fxi - float(labelMatrix[i])
            if labelMatrix[i] * Ei < -toler and alpha[i] < C or labelMatrix[i] * Ei > toler and alpha[i] > 0:
                j = selectJrand(i,m)
                fxj = float(multiply(alpha,labelMatrix).T * (dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fxj - float(labelMatrix[j])
                alphaIOld = alpha[i].copy()
                alphaJOld = alpha[j].copy()
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0,alpha[j] - alpha[i])
                    H = min(C,C+alpha[j]-alpha[i])
                else:
                    L = max(0,alpha[i]+alpha[j] - C)
                    H = min(C,alpha[j]+alpha[i])
                if L==H:
                    print "L==H"
                    continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print "eta >= 0"
                    continue
                alpha[j] -= labelMatrix[j]*(Ei-Ej)/eta
                alpha[j] = clipAlpha(alpha[j],H,L)
                if abs(alpha[j] - alphaJOld) < 0.00001 :
                    print "j not move enough"
                    continue
                alpha[i] += labelMatrix[j]*labelMatrix[i]*(alphaJOld - alpha[j])
                b1 = b - Ei -labelMatrix[i]*(alpha[i] - alphaIOld)*dataMatrix[i,:]*dataMatrix[i,:].T \
                -labelMatrix[j]*(alpha[j]-alphaJOld)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej -labelMatrix[i]*(alpha[i] - alphaIOld)*dataMatrix[i,:]*dataMatrix[j,:].T \
                -labelMatrix[j]*(alpha[j]-alphaJOld)*dataMatrix[j,:]*dataMatrix[j,:].T
                if alpha[i] > 0 and alpha[i] < C:
                    b = b1
                elif alpha[j] > 0 and alpha[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphapairChanged +=1
                print "iter: %d i:%d,oairs changed %d" %(iter,i,alphapairChanged)
        if alphapairChanged == 0:
            iter +=1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return b,alpha



def main():
    data,label = loadDataSet('testSet.txt')
    print data
    print label
    b,alpha = smoSimple(data,label,0.6,0.001,40)
    print b
    print alpha
    for i in range(100):
        if alpha[i]>0:
            print data[i],label[i]
if __name__ == '__main__':
    sys.exit(main())