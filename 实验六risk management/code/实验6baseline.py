'''
risk model
'''

import lightgbm as lgb
import pandas as pd
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import math
import matplotlib.pyplot as plt

def myLR(train_set_x,train_set_y,test_set_x,learning_rate = 0.1,n_epochs = 15,batch_size = 20,
    linear_search = 0,lamda = 0.01):
    n_in=train_set_x.shape[1]
    n_out=2

    n_train_batches = train_set_x.shape[0] // batch_size
    n_test_batches = test_set_x.shape[0] // batch_size

    W = numpy.zeros((n_in, n_out), dtype='float64')
    b = numpy.zeros((n_out,), dtype='float64')
    p_y_given_x = numpy.zeros((batch_size, n_out), dtype='float64')
    g_W = numpy.zeros((n_in, n_out), dtype='float64')
    g_b = numpy.zeros((n_out,), dtype='float64')

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    pred = []
    epoch = 0
    while (epoch < n_epochs):
        epoch = epoch + 1
        print(epoch)
        for minibatch_index in range(n_train_batches):
            index = minibatch_index
            x = train_set_x[index * batch_size: (index + 1) * batch_size]
            y = train_set_y[index * batch_size: (index + 1) * batch_size]
            # 计算分布函数
            thi = numpy.matmul(x, W)
            for i in range(batch_size):
                p_y_given_x[i] = thi[i] + b
            p_y_given_x = numpy.exp(p_y_given_x)
            sum = numpy.sum(p_y_given_x, axis=1)
            for i in range(batch_size):
                p_y_given_x[i] = numpy.divide(p_y_given_x[i], sum[i])
            y_pred = numpy.argmax(p_y_given_x, axis=1)

            # 计算梯度
            y_is_j = []
            for i in range(batch_size):
                nowyisj = []
                for j in range(n_out):
                    if y[i] == j:
                        nowyisj.append(1)
                    else:
                        nowyisj.append(0)
                y_is_j.append(nowyisj)
            umm = numpy.asarray(y_is_j)
            coef = umm - p_y_given_x
            for a in range(batch_size):
                for i in range(n_in):
                    for j in range(n_out):
                        g_W[i][j] -= x[a][i] * coef[a][j]
            g_W /= batch_size
            g_W += lamda * W
            g_b = -1.0 * numpy.mean(coef, axis=0) + lamda * b
            if linear_search == 0:
                W = W - learning_rate * g_W
                b = b - learning_rate * g_b
            else:
                a = 1.0
                c = 0.5
                phy = 0.5
                m = numpy.sum(numpy.square(g_W))
                fx = 0.0
                for i in range(batch_size):
                    fx -= math.log(p_y_given_x[i][y[i]])
                fx /= batch_size
                while (1):
                    thi = numpy.matmul(x, W - a * g_W)
                    for i in range(batch_size):
                        p_y_given_x[i] = thi[i] + b
                    p_y_given_x = numpy.exp(p_y_given_x)
                    sum = numpy.sum(p_y_given_x, axis=1)
                    for i in range(batch_size):
                        p_y_given_x[i] = numpy.divide(p_y_given_x[i], sum[i])
                    nowfx = 0.0
                    for i in range(batch_size):
                        nowfx -= math.log(p_y_given_x[i][y[i]])
                    nowfx /= batch_size
                    if nowfx <= fx - a * c * m:
                        break
                    else:
                        a = a * phy
                W = W - a * g_W
                a = 1.0
                c = 0.5
                phy = 0.5
                m = numpy.sum(numpy.square(g_b))
                while (1):
                    thi = numpy.matmul(x, W)
                    for i in range(batch_size):
                        p_y_given_x[i] = thi[i] + b - a * g_b
                    p_y_given_x = numpy.exp(p_y_given_x)
                    sum = numpy.sum(p_y_given_x, axis=1)
                    for i in range(batch_size):
                        p_y_given_x[i] = numpy.divide(p_y_given_x[i], sum[i])
                    nowfx = 0.0
                    for i in range(batch_size):
                        nowfx -= math.log(p_y_given_x[i][y[i]])
                    nowfx /= batch_size
                    if nowfx <= fx - a * c * m:
                        break
                    else:
                        a = a * phy
                b = b - a * g_b

    pred.clear()
    for index in range(n_test_batches):
        x = test_set_x[index * batch_size: (index + 1) * batch_size]
        # 计算分布函数
        thi = numpy.matmul(x, W)
        for i in range(batch_size):
            p_y_given_x[i] = thi[i] + b
        p_y_given_x = numpy.exp(p_y_given_x)
        sum = numpy.sum(p_y_given_x, axis=1)
        for i in range(batch_size):
            p_y_given_x[i] = numpy.divide(p_y_given_x[i], sum[i])
            pred.append(p_y_given_x[i][1])
    return pred

class RiskModel():
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.train, self.test, self.param, self.final_test= self.__construct_dataset()
        self.feature_name = [i for i in self.train.columns if i not in ['Y']]
        print('train set:', self.train.shape, ', ', 'test set:', self.test.shape)
        print(self.feature_name)
        self.lgb_train = lgb.Dataset(data=self.train[self.feature_name],
                                     label=self.train['Y'],
                                     feature_name=self.feature_name)
        self.lgb_test = lgb.Dataset(data=self.test[self.feature_name],
                                    label=self.test['Y'],
                                    feature_name=self.feature_name)
        self.evals_result = {}
        self.gbm = None
        self.LR=None
        self.GBDT=None

    def __construct_dataset(self):
        train = pd.read_csv(self.data_path + 'train.csv')
        test = pd.read_csv(self.data_path + 'test.csv')
        final_test = pd.read_csv(self.data_path+'final_test.csv')
        train = train.astype('float')
        test = test.astype('float')
        final_test = final_test.astype('float')

        param = dict()
        param['objective'] = 'binary'
        param['boosting_type'] = 'gbdt'
        param['metric'] = 'auc'
        param['verbose'] = 0
        param['learning_rate'] = 0.1
        param['max_depth'] = -1
        param['feature_fraction'] = 0.8
        param['bagging_fraction'] = 0.8
        param['bagging_freq'] = 1
        param['num_leaves'] = 15
        param['min_data_in_leaf'] = 64
        param['is_unbalance'] = False
        param['verbose'] = -1

        return train, test, param, final_test

    def fit(self):
        self.gbm = lgb.train(self.param,
                             self.lgb_train,
                             early_stopping_rounds=10,
                             num_boost_round=1000,
                             evals_result=self.evals_result,
                             valid_sets=[self.lgb_train, self.lgb_test],
                             verbose_eval=1)

        train_y=self.train['Y'].tolist()
        train_x=self.train.drop("Y",axis=1).values.tolist()
        self.LR=LogisticRegression(penalty='l1',solver='liblinear',max_iter=100,tol=0.01)
        self.GBDT=GradientBoostingClassifier(learning_rate=0.1,n_estimators=100)
        self.LR.fit(train_x, train_y)
        self.GBDT.fit(train_x,train_y)

    def myAUC(self,prob,labels):
        f = list(zip(prob,labels))
        rank = [x2 for x1,x2 in sorted(f,key=lambda x:x[0])]
        rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
        cnt_pos = 0
        cnt_neg = 0
        for i in range(len(labels)):
            if(labels[i]==1):
                cnt_pos+=1
            else:
                cnt_neg+=1
        return (sum(rankList)- (cnt_pos*(cnt_pos+1))/2)/(cnt_pos*cnt_neg)

    def evaluate(self):
        train_y = self.train['Y'].tolist()
        train_x = self.train.drop("Y", axis=1).values.tolist()
        test_x = self.test.drop("Y", axis=1).values.tolist()
        pred_LR = self.LR.predict_proba(test_x)
        pred_GBDT  = self.GBDT.predict_proba(test_x)
        pred_myLR = myLR(numpy.asarray(train_x),numpy.asarray(train_y),numpy.asarray(test_x))
        test_label = self.test['Y']
        final_test_label = self.gbm.predict(self.final_test)
        out=numpy.c_[numpy.arange(50000,50000+final_test_label.shape[0],1),final_test_label]
        df = pd.DataFrame(out)
        df.columns = ['id', 'pre']
        df.to_csv("./result.csv", index=False)
        del self.test["Y"]
        prob_label = self.gbm.predict(self.test)

        # 画特征重要性的柱状图的代码
        # importances = self.GBDT.feature_importances_
        # indices = numpy.argsort(importances)[::-1]
        # indices = indices[0:15]
        # num_features = indices.shape[0]
        # m = {}
        # for i in range(len(self.feature_name)):
        #     m.setdefault(self.feature_name[i], 0)
        #     m[self.feature_name[i]] = self.LR.coef_[0][i]
        # m=sorted(m.items(), key=lambda x: x[1], reverse=True)
        # m=m[0:15]
        # print(m)
        # key=[]
        # value=[]
        # for i in range(15):
        #     key.append(m[i][0])
        #     value.append(m[i][1])
        # plt.figure()
        # plt.title("Feature importances")
        # plt.bar(range(15), value, color="g", align="center")
        # plt.xticks(range(15), key, rotation='45')
        # plt.xlim([-1, 15])
        # plt.show()
        
        auc= self.myAUC(prob_label,test_label)
        auc_GBDT= self.myAUC(pred_GBDT[:,1],test_label)
        auc_LR= self.myAUC(pred_LR[:,1],test_label)
        auc_myLR = self.myAUC(pred_myLR,test_label)
        return auc,auc_GBDT,auc_LR,auc_myLR


if __name__ == "__main__":
    MODEL = RiskModel(data_path='./')
    MODEL.fit()
    auc,auc_GBDT,auc_LR,auc_myLR=MODEL.evaluate()
    print('eval auc:')
    print("auc_gbm:", auc)
    print("auc_GBDT:", auc_GBDT)
    print("auc_LR", auc_LR)
    print("auc_myLR", auc_myLR)
