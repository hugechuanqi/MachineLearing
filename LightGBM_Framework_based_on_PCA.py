# coding: utf-8
## 研究方向：网络入侵检测（Network Intrusion Detection）
## python使用库：pandas, sklearn, matplotlib, numpy, lightGBM, time
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.metrics import confusion_matrix
from sklearn.metrics import classification
import seaborn as sns 
import time
import gc
from sklearn.decomposition import PCA

class NetworkIntrusionDetection:
    """ 网络入侵检测模型简历和检验的基本方法
    """
    def __init__(self, labels_dict, params , classFlag=2):
        self.features_ = None
        self.labels_dict_ = labels_dict
        self.params_ = params
        self.classFlag_ = classFlag
        self.model_ = None

    def loadData(self, path, Features):
        """ 加载数据集，返回DataFrame
        """
        self.features_ = Features
        dataSet_df = pd.read_csv(path, names=self.features_)
        return dataSet_df
    
    def BinaryClassify(self, dataSet_df):
        """ 修改分类是进行二分类还是多分类，返回DataFrame
        """
        df = dataSet_df.copy()
        if self.classFlag_ == 2:
            normal_index = df[df['label'].isin(self.labels_dict_["normal"])].index
            abnormal_index = df[~df['label'].isin(self.labels_dict_["normal"])].index
            df.loc[normal_index, 'label'] = 'normal'
            df.loc[abnormal_index, 'label'] = 'abnormal'
        else:
            df.loc[df['label'].isin(self.labels_dict_["normal"]), 'label'] = 'normal'
            df.loc[df['label'].isin(self.labels_dict_["Dos"]), 'label'] = 'Dos'
            df.loc[df['label'].isin(self.labels_dict["Probing"]), 'label'] = 'Probing'
            df.loc[df['label'].isin(self.labels_dict["R2L"]), 'label'] = 'R2L'
            df.loc[df['label'].isin(self.labels_dict["U2R"]), 'label'] = 'U2R'
        print("label标签分类包括：", set(df['label'] .values))
        for labelType in set(df['label'].values):
            print("-----类别", labelType, "数量为：", df[df['label']==labelType].shape)
        return df

    def is_number(self, dataSeries):
        """ 判断是否为数值型数据，返回bool
        """
        try:
            s = str(int(dataSeries.iloc[0]))
            if s.isnumeric():
                return True
            else:
                return False
        except:
            return False

    def featureProcessing(self, dataSet_df):
        """ 特征工程，返回DataFrame
        """
        df = dataSet_df.copy()
        print(len(self.features_))
        ## 1、修改数据标签
        df  = self.BinaryClassify(df)
 
        ## 2、数据清洗之删除固定特征
        for feature in self.features_:
            if len(set(df[feature].values))==1:
                print("删除特征：", feature)
                df.drop([feature], axis=1, inplace=True)
        self.features_ = df.columns
        print(len(self.features_))

        ## 3、特征编码之类别编码
        for feature in self.features_:
            if self.is_number(df[feature]):
                df.loc[:, feature] = df.loc[:,feature].astype(dtype=float)    
            else:
                print("类别编码：", feature)
                lbl = preprocessing.LabelEncoder()
                lbl.fit(df.loc[:, feature].values)
                df.loc[:, feature] = lbl.transform(df.loc[:,feature].values)
        return df

    def dataSet_Partition(self, dataSet_df_new, test_size_=0.1):
        """ 对数据集进行划分，划分比例为验证集占test_size，或者划分比例为训练集占train_size
        """
        df  = dataSet_df_new.copy()
        trainDataSet, valDataSet = train_test_split(df, test_size=test_size_, random_state=1) 
        return trainDataSet, valDataSet

    def modelTraining(self, trainDataSet, valDataSet):
        """ 对数据集进行模型训练，返回模型参数model
        """
        train_X, train_y = trainDataSet.loc[:, self.features_[:-2]],  trainDataSet.loc[:, self.features_[-1]].values
        val_X, val_y =  valDataSet.loc[:, self.features_[:-2]],  valDataSet.loc[:, self.features_[-1]].values

        lgtrain = lgb.Dataset(train_X, label=train_y)
        lgval = lgb.Dataset(val_X, label=val_y)
        model = lgb.train(params = self.params_, train_set=lgtrain, num_boost_round=1000, valid_sets=[lgval], early_stopping_rounds=1000, verbose_eval=100)
        self.model_ = model
     
    def modelPrediction(self, testData):
        """ 模型标签预测
        """
        test_X, test_y =  testData.loc[:, self.features_[:-1]],  testData.loc[:, self.features_[-1]].values
        pred_label = self.model_.predict(test_X, num_iteration=self.model_.best_iteration)
        return pred_label

class ResultVisualization:
    """ 结果可视化类
    """
    def __init__(self, path):
        """
        path_：存储路径
        flag_：测试集标记
        pred_label_：预测标签结果
        actual_label_：实际标签结果
        dataSet_：待测试数据集
        length_：测试集长度
        """
        self.path_ = path
        self.name_ = ""
        self.pred_label_ = None
        self.actual_label_ = None
        self.dataSet_ = None
        self.length_ = 0
    
    def load_pred(self, pred_label, dataSet):
        """ 加载模型预测结果和数据集
        """
        prediction_y = []
        for item in pred_label:
            prediction_y.append(np.argmax(item))
        self.pred_label_ = prediction_y
        self.actual_label_ = dataSet["label"].values
        self.dataSet_ = dataSet
        self.length_ = dataSet.shape[0]
    
    def modelCheck(self, pred_label, dataSet, name):
        self.load_pred(pred_label, dataSet)     # 加载数据集
        self.name_ = name

        cm = confusion_matrix(self.actual_label_, self.pred_label_)
        print("混淆矩阵为：\n", cm)

        ##检测率##
        detection_Rate = (cm[0,0]+cm[1,1])/(cm.sum())
        detectionNormal_Rate = cm[1,1] / cm[:,1].sum()    # normal检测率
        detectionAbnormal_Rate = cm[0,0] / cm[:,0].sum()   #异常检测率
        
        print("\n总体识别率（准确率）为", detection_Rate)
        print("normal识别率（准确率）为", detectionNormal_Rate)
        print("abnormal异常识别率（准确率）为", detectionAbnormal_Rate)

        cm_df = pd.DataFrame(cm)
        sns.heatmap(cm_df, annot=True)

    def get_case(self):
        self.dataSet_.loc[:, "pred_label"] = pd.Series(self.pred_label_)
        self.dataSet_.loc[:, "result"] = self.dataSet_.loc[:, "label"] - self.dataSet_.loc[:,"pred_label"]
        self.dataSet_.to_csv(self.path_ + self.name_ + ".csv", index=False)     #index_label=False 表示列索引字段不打印出来

    def labelCount(self, dataSet):
        plt.savefig(save_path)
    def plotResultCurve(self, pred_label):
        plt.savefig(save_path)
    def plotFeatureImportance(self, model):
        plt.savefig(save_path)


## lightGBM框架主函数
def lightgbm_main(Path, Features, Labels, lgb_train_params, important_Features):
    """ lightGBM框架主函数
    """
    ## 1、模型预处理
    classFlag = 2   #表示为二分类
    NID = NetworkIntrusionDetection(Labels, lgb_train_params, classFlag)
    dataSet_df = NID.loadData(Path, Features)
    print(dataSet_df.head())
    dataSet_df_new = NID.featureProcessing(dataSet_df)

    # ## 2、特征降维
    # x = dataSet_df_new.loc[:, :-1]
    # y = dataSet_df_new.loc[:, -1]
    # pca = PCA(n_components = 19)        # 将特征降至19维
    # reduced_x = pca.fit_transform(x)
    dataSet_df_new = dataSet_df_new[important_Features]

    ## 3、模型训练
    test_size = 0.1
    trainDataSet, valDataSet = NID.dataSet_Partition(dataSet_df_new, test_size)
    start_time = time.time()
    NID.modelTraining(trainDataSet, valDataSet)
    end_time = time.time()
    print("模型训练时间为：", end_time - start_time)
    with open("/home/huge/Documents/python_code/paper_code/paper1/lgb_model.txt", "w") as fr:
        fr.write(str(NID.model_))
        print("模型参数存储完成")
        fr.close()
    pred_y = NID.modelPrediction(valDataSet)
    # del trainDataSet    #释放内存
    # gc.collection()

    ## 4、模型检验
    ### 4.1 验证集检验
    store_path = "/home/huge/Documents/python_code/paper_code/input_data/"
    RV = ResultVisualization(store_path)
    print("+++++验证集准确率结果：\n")
    name = "valDataSet"
    RV.modelCheck(pred_y, valDataSet, name)

    ### 4.2 测试集1和测试集2
    test_size2=0.5
    testDataSet1, testDataSet2 = NID.dataSet_Partition(dataSet_df_new, test_size2)
    pred_y1 = NID.modelPrediction(testDataSet1)
    pred_y2 = NID.modelPrediction(testDataSet2)
    print("\n+++++第一个测试集的准确率结果：\n")
    RV.modelCheck(pred_y1, testDataSet1, "testDataSet1")
    print("\n+++++第二个测试集的准确率结果：\n")
    RV.modelCheck(pred_y2, testDataSet2, "testDataSet2")


    ### 4.3 测试集3和测试集4
    test_Path2 = "/home/huge/Documents/python_code/paper_code/input_data/corrected"
    dataSet_df2 = NID.loadData(test_Path2, Features)
    dataSet_df_new2= NID.featureProcessing(dataSet_df2)
    test_size3 = 0.5
    testDataSet3, testDataSet4 = NID.dataSet_Partition(dataSet_df_new2, test_size3)
    pred_y3 = NID.modelPrediction(testDataSet3)
    pred_y4 = NID.modelPrediction(testDataSet4)
    print("\n+++++第三个测试集的准确率结果：\n")
    RV.modelCheck(pred_y3, testDataSet3, "testDataSet3")
    RV.get_case()
    print("\n+++++第四个测试集的准确率结果：\n")
    RV.modelCheck(pred_y4, testDataSet4, "testDataSet4")
    RV.get_case()


if __name__ == "__main__":
    Path = "/home/huge/Documents/python_code/paper_code/input_data/kddcup.data_10_percent_corrected"
    Features=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent', 'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root', 'num_file_creations','num_shells', 'num_access_files','num_outbound_cmds','is_hot_login','is_guest_login', 'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate', 'diff_srv_rate','srv_diff_host_rate', 'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate', 'dst_host_srv_serror_rate','dst_host_rerror_rate','st_host_srv_rerror_rate', 'label']
    important_Features = ['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes','urgent', 'count', 'srv_count', 'same_srv_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_same_src_port_rate', 'label']
    labels_dict = {
        "normal":["normal."],
        "Dos":['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.', 'apache2.',                     'mailbomb.', 'processtable.', 'udpstorm.'],
        "Probing":['ipsweep.', 'nmap.', 'portsweep.', 'satan.', 'mscan.', 'saint.'],
        "R2L":['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.', 'snmpgetattack.', 'named.', 'sendmail.', 'snmpgeattack.', 'snmpguess.', 'worm.', 'xlock.', 'xsnoop.'],
        "U2R":['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.', 'httptunnel.', 'ps.', 'sqlattack.', 'xterm.']
    }
    lgb_train_params = {
        # "objective": "binary",
        "objective": "multiclass",
        "num_class": 2,
        "learning_rate": 0.1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.2,
        "max_depth": 10,
        "num_leaves": 50,
    }

    ## 进行模型训练（包括对所有特征还是13个重要特征）
    lightgbm_main(Path, Features, labels_dict, lgb_train_params, important_Features)
    

