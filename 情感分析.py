import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
def read_reviews(path):
    reviews=[]
    labels=[]
    for dirpath,dirnames,filenames in os.walk(path):
        for filename in filenames:
            rating=int(filename.split('_')[1].split('.')[0])
            if rating<=4:
                labels.append(0)
            elif rating>=7:
                labels.append(1)
            file_path=os.path.join(dirpath,filename)
            with open(file_path,'r',encoding='UTF-8') as file:
                review=file.read()
            reviews.append(review)
    return reviews,labels
def preprocess_reviews(reviews):
    stop_words = set(stopwords.words('english'))#获取英语中的停用词
    lemmatizer = WordNetLemmatizer() #还原词形    
    reviews=[nltk.word_tokenize(review) for review in reviews]#分词器
    reviews=[[lemmatizer.lemmatize(word) for word in review if word not in stop_words and word.isalpha()]for review in reviews]
    return reviews


"""读取数据内容并进行预处理"""
train_reviews_pos,train_labels_pos=read_reviews("D:\\aclImdb\\train\\pos")
train_reviews_neg,train_labels_neg=read_reviews("D:\\aclImdb\\train\\neg")
train_reviews=train_reviews_pos+train_reviews_neg
train_labels=train_labels_neg+train_labels_pos
test_reviews_pos,test_labels_pos=read_reviews("D:\\aclImdb\\test\\pos")
test_reviews_neg,test_labels_neg=read_reviews("D:\\aclImdb\\test\\neg")
test_reviews=test_reviews_pos+test_reviews_neg
test_labels=test_labels_neg+test_labels_pos
train_reviews = preprocess_reviews(train_reviews)
test_reviews = preprocess_reviews(test_reviews)


"""利用朴素贝叶斯分类器进行训练"""
train_reviews_text=[' '.join(train_review) for train_review in train_reviews]
test_reviews_text=[' '.join(test_review) for test_review in test_reviews]
vectorizer=CountVectorizer()
train_feartures=vectorizer.fit_transform(train_reviews_text)
test_feartures=vectorizer.transform(test_reviews_text)
model=MultinomialNB()
model.fit(train_feartures,train_labels)
predictions=model.predict(test_feartures)
accuracy=accuracy_score(test_labels,predictions)
print(accuracy)


"""绘制出混淆矩阵的热力图"""
cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix', size = 15)
plt.show()


"""绘制ROC曲线"""
fpr, tpr, thresholds = roc_curve(test_labels, model.predict_proba(test_feartures)[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
