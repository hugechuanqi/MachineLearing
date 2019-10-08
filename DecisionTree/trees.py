'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    # 总共5个样本，2个特征('no surfacing','flippers')，一个类别列(yes, no)
    dataSet = [[1, 1, "青年", "本科生",'yes'],
               [1, 0, "少年", "高中生",'yes'],
               [1, 0, "少年", "本科生", 'no'],
               [0, 0, "青年", "本科生",'no'],
               [0, 1, "青年", "高中生", 'no'],
               [0, 1, "老年", "本科生", 'no'],
               [0, 0, "青年", "高中生", 'yes'],
               [1, 1, "青年", "高中生", 'yes'],
               [0, 0, "少年", "本科生", 'yes']]
    labels = ['first','second','third','forth']
    #change to discrete values
    return dataSet, labels

## 计算信息增益
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries       # 计算概率
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt
    
## 划分数据集，将选择的特征数据单独取出，返回剩余的数据集（同理，标签列也应该在此处删除）
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            chooseFeatureValue = featVec[axis]     #chop out axis used for splitting
            # print("    待移除特征数据：", chooseFeatureValue)
            reducedFeatVec = featVec[:axis] + featVec[axis+1:]
            retDataSet.append(reducedFeatVec)
    return retDataSet
 
## 选择最优特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:    # value表示每个特征下取的值
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            # print("划分出来的子集为：", subDataSet, "子集长度为：", len(subDataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

## 
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

## 创建树结构
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    print("第一个标签种类数为：", classList.count(classList[0]), "标签种类数为：", len(classList))
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    print("   最优的特征为：第 ", bestFeat+1, " 个", "标签为：", labels[bestFeat])
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])   # 找到最优特征后，取出其中的最优特征的特征名称

    featValues = [example[bestFeat] for example in dataSet]     # 取出当前特征列的所有数据
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        print("   ", bestFeatLabel, value, subLabels)
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
## 决策树生成主函数
def DecisionTree_mian():
    dataSet, labels = createDataSet()
    print(dataSet, labels)
    DecesionTree = createTree(dataSet, labels)
    print(DecesionTree)

if __name__ == "__main__":
    DecisionTree_mian()