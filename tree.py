import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
## 定义 决策树模型
class tree():
    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train = x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test = y_test
    def modelfit(self):

        clf = DecisionTreeClassifier(criterion='entropy')
        # 在训练集上训练决策树模型
        clf.fit(self.x_train, self.y_train)
        #%% 在训练集和测试集上利用训练好的模型进行预测
        self.train_predict = clf.predict(self.x_train)
        self.test_predict = clf.predict(self.x_test)
    def getaccuracy(self):

        ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
        print('The accuracy of the train_DecisionTree is:',metrics.accuracy_score(self.y_train,self.train_predict))
        print('The accuracy of the test_DecisionTree is:',metrics.accuracy_score(self.y_test,self.test_predict))
    def show_confusion(self):

        ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
        confusion_matrix_result = metrics.confusion_matrix(self.test_predict,self.y_test)
        print('The confusion matrix result:\n',confusion_matrix_result)
        # 利用热力图对于结果进行可视化
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()