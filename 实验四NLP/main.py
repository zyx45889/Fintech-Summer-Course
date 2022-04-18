import tushare as ts
import pandas
import jieba
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba.analyse
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import matplotlib.pyplot as plt


##### 读取数据 #####

ts.set_token("a6dae538a760f0b9e39432c1bff5e50a1c462a1a087e994dae18fa04")
pro = ts.pro_api()
feature=['软件服务','元器件','电气设备','化工原料','专用机械','通信设备']
data1 = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
data2 = pro.stock_company(exchange='SZSE', fields='ts_code,business_scope')
with open('stop_words.txt', 'r', encoding='utf8') as f:
    file = f.read().split('\n')
stop_words=set(file)


##### 数据预处理 #####

data=pandas.merge(data1,data2)
data=data.dropna()
data=data[data['industry'].isin(feature)]
for i in range(len(feature)):
    data.industry[data['industry']==feature[i]]=i
scope=data['business_scope']
industry=data['industry']


##### 分词 #####

def check(str):
    mark=False
    for i in str:
        if u'\u4e00' <= i <= u'\u9fff':
            mark=True
    return mark

def remove_stop_word(words):
    temp = list(words)
    words_list=[]
    for i in temp:
        if i in stop_words:
            continue
        elif check(i)==False:
            continue
        elif i.isdigit():
            continue
        words_list.append(i)
    return words_list

word_list=[]
for str in scope:
    temp1=remove_stop_word(jieba.cut(str))
    # temp2=remove_stop_word(jieba.analyse.extract_tags(str))
    word_list.append(" ".join(i for i in temp1))#+" "+" ".join((j for j in temp2)))


##### 文本数值化 #####

v = TfidfVectorizer()
vector=v.fit_transform(word_list)
key=v.get_feature_names()
target=[]
for i in industry:
    target.append(i)


##### 训练分类器 #####

kf = KFold(n_splits=12,shuffle=True)
accuracy=0
avg=0
for train_index, test_index in kf.split(target):
    X_train, X_test = vector[train_index], vector[test_index]
    y_train, y_test = [],[]
    for i in train_index:
        y_train.append(target[i])
    for i in test_index:
        y_test.append(target[i])
    # clf = tree.DecisionTreeClassifier()
    # clf.fit(X_train, y_train)
    # pred = clf.predict(X_test)
    svc = svm.SVC(kernel='sigmoid')
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    count = 0
    for l, r in zip(pred, y_test):
        if l == r:
            count += 1
    avg=avg+count/len(y_test)
    if(count/len(y_test)>accuracy):
        accuracy=count/len(y_test)
    print(classification_report(y_test, pred, target_names=feature))
print("max accuracy:",accuracy)
print("average accuracy:",avg/12)

#### 报告中展示的其他部分，如多种分类器的效果测试，训练集大小对分类正确率的影响作图等没有在这份代码中展示
#### 这里保留的是最佳的分类器搭配，输出要求的评估参数和最佳/平均正确率