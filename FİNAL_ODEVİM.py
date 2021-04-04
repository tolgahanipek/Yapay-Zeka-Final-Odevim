#KÜTÜPHANELER
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import matplotlib.image as mpimg




# DOSYAYI OKUMA
dataset=pd.read_csv("data.csv")
print(dataset.head(10))

print(dataset.shape)

print(dataset.info())

#1.SORUNUN CEVABI
plt.figure(figsize=(10,10))
plt.title("-HASTALIK VE SAĞLAMLIK GRAFİĞİ-")
sns.countplot(x="target",data=dataset)
plt.xlabel("HASTA VEYA SAĞLAM OLMA DURUMU")
plt.ylabel("KİŞİ SAYISI")
plt.show()
#2.SORUNUN CEVABI
plt.figure(figsize=(10,10))
plt.title("CİNSİYETE GÖRE SAĞLIKLI VE HASTA SÜTUN GRAFİĞİ")
sns.countplot(y="target", hue="sex", data=dataset, palette="Greens_d");
plt.xlabel("KİŞİ SAYISI")
plt.ylabel("HASTA VEYA SAĞLAM OLMA DURUMU")
plt.show()
#3.SORUNUN CEVABI
plt.figure(figsize=(10,10))
plt.title("YAŞ DAĞILIMI SÜTUN GRAFİĞİ")
sns.countplot(x="age",data=dataset);
plt.xlabel("YAŞ")
plt.ylabel("KİŞİ SAYISI")
plt.show()
#4.SORUNUN CEVABI
plt.figure(figsize=(10,10))
plt.title("HASTA OLANLARIN YAŞ DAĞILIM GRAFİĞİ")
hs=dataset[dataset['target']==1]
sns.distplot(hs["age"])
plt.xlabel("YAŞ")
plt.show()
#5. VE 6. SORULARIN CEVAPLARI
from sklearn.model_selection import train_test_split
X=dataset.iloc[:,0:13]
y=dataset.iloc[:,13]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.20)
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
print(X_test)
#7.SORUNUN CEVABI LOGİSTİC REGRESSİON
print("-------LOJİSTİK REGRESYON İLE SINIFLANDIRMA")
logmodel=lr(max_iter=200)
logmodel.fit(X_train,y_train)
logmodel.coef_
pred=logmodel.predict(X_test)

print("PREDICT:")
print(pred)

print(X_test.shape)
print("KARMAŞIKLIK MATRİSİ")
m=confusion_matrix(y_test,pred)
print(m)

print("KARMAŞIKLIK MATRİSİ GRAFİĞİ")
plt.figure(figsize=(10,10))
sns.heatmap(m,annot=True,annot_kws={"size": 20})
plt.xlabel("PREDİCTED")
plt.ylabel("ACTUAL")
plt.show()

print("SINIFLANDIRMA RAPORU")
print(classification_report(y_test,pred))

print("ACCURACY ORANI:")
print((m[0,0]+m[1,1])/(m[0,0]+m[0,1]+m[1,0]+m[1,1]))

print("PRECISION ORANI:")
print(m[0,0]/(m[0,0]+m[1,0]))

print("RECALL ORANI:")
print(m[0,0]/(m[0,0]+m[0,1]))

print("SENSİTİVİTY ORANI:")
print(m[0,0]/(m[0,0]+m[0,1]))

print("SPECİFİCİTY ORANI")
print(m[1,1]/(m[1,0]+m[1,1]))

#8.SORUNUN CEVABI K-NN
print("---------------KNN İLE SINIFLANDIRMA")
classifier=KNeighborsClassifier(n_neighbors=11,metric='euclidean')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print("PREDICT:")
print (y_pred)
cm=confusion_matrix(y_test,y_pred)
print("KARMAŞIKLIK MATRİSİ")
print (cm)
print("KARMAŞIKLIK MATRİSİ GRAFİĞİ")
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,annot_kws={"size": 20})
plt.xlabel("PREDİCTED")
plt.ylabel("ACTUAL")
plt.show()

print("SINIFLANDIRMA RAPORU")
print(classification_report(y_test,y_pred))

print("ACCURACY ORANI:")
print((cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]))

print("PRECISION ORANI:")
print(cm[0,0]/(cm[0,0]+cm[1,0]))

print("RECALL ORANI:")
print(cm[0,0]/(cm[0,0]+cm[0,1]))

print("SENSİTİVİTY ORANI:")
print(cm[0,0]/(cm[0,0]+cm[0,1]))

print("SPECİFİCİTY ORANI")
print(cm[1,1]/(cm[1,0]+cm[1,1]))

#9.SORUNUN CEVABI NAİVE BAYES
print("---------------NAİVE BAYES İLE SINIFLANDIRMA")

from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)
z_pred = nvclassifier.predict(X_test)
print("PREDİCT:")
print(z_pred)
y_compare = np.vstack((y_test,z_pred)).T
print("------------------")
print(y_compare[:13,:])
Cm = confusion_matrix(y_test, z_pred)
print("KARMAŞIKLIK MATRİSİ")
print(Cm)
print("KARMAŞIKLIK MATRİSİ GRAFİĞİ")
plt.figure(figsize=(10,10))
sns.heatmap(Cm,annot=True,annot_kws={"size": 20})
plt.xlabel("PREDİCTED")
plt.ylabel("ACTUAL")
plt.show()

print("SINIFLANDIRMA RAPORU")
print(classification_report(y_test,z_pred))

print("ACCURACY ORANI:")
print((Cm[0,0]+Cm[1,1])/(Cm[0,0]+Cm[0,1]+Cm[1,0]+Cm[1,1]))

print("PRECISION ORANI:")
print(Cm[0,0]/(Cm[0,0]+Cm[1,0]))

print("RECALL ORANI:")
print(Cm[0,0]/(Cm[0,0]+Cm[0,1]))

print("SENSİTİVİTY ORANI:")
print(Cm[0,0]/(Cm[0,0]+Cm[0,1]))

print("SPECİFİCİTY ORANI")
print(Cm[1,1]/(Cm[1,0]+Cm[1,1]))

#10.SORUNUN CEVABI DESICION_TREE
print("---------------DESICION TREE İLE SINIFLANDIRMA")
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
t_pred = dt.predict(X_test)
CM=confusion_matrix(y_test,t_pred)
print("KARMAŞIKLIK MATRİSİ")
print(CM)

print("KARMAŞIKLIK MATRİSİ GRAFİĞİ")
plt.figure(figsize=(10,10))
sns.heatmap(CM,annot=True,annot_kws={"size": 20})
plt.xlabel("PREDİCTED")
plt.ylabel("ACTUAL")
plt.show()

print("SINIFLANDIRMA RAPORU")
print(classification_report(y_test,t_pred))

print("ACCURACY ORANI:")
print((CM[0,0]+CM[1,1])/(CM[0,0]+CM[0,1]+CM[1,0]+CM[1,1]))

print("PRECISION ORANI:")
print(CM[0,0]/(CM[0,0]+CM[1,0]))

print("RECALL ORANI:")
print(CM[0,0]/(CM[0,0]+CM[0,1]))

print("SENSİTİVİTY ORANI:")
print(CM[0,0]/(CM[0,0]+CM[0,1]))

print("SPECİFİCİTY ORANI")
print(CM[1,1]/(CM[1,0]+CM[1,1]))

#11.SORUNUN CEVABI YAPAY SİNİR AĞI
print("---------------YAPAY SİNİR AĞI İLE SINIFLANDIRMA")
MLP_classifier = MLPClassifier(max_iter=1000)
MLP_classifier.fit(X_train, y_train)
s_pred = MLP_classifier.predict(X_test)
Mr=confusion_matrix(y_test,s_pred)
print("KARMAŞIKLIK MATRİSİ")
print(Mr)

print("KARMAŞIKLIK MATRİSİ GRAFİĞİ")
plt.figure(figsize=(10,10))
sns.heatmap(Mr,annot=True,annot_kws={"size": 20})
plt.xlabel("PREDİCTED")
plt.ylabel("ACTUAL")
plt.show()

print("SINIFLANDIRMA RAPORU")
print(classification_report(y_test,s_pred))

print("ACCURACY ORANI:")
print((Mr[0,0]+Mr[1,1])/(Mr[0,0]+Mr[0,1]+Mr[1,0]+Mr[1,1]))

print("PRECISION ORANI:")
print(Mr[0,0]/(Mr[0,0]+Mr[1,0]))

print("RECALL ORANI:")
print(Mr[0,0]/(Mr[0,0]+Mr[0,1]))

print("SENSİTİVİTY ORANI:")
print(Mr[0,0]/(Mr[0,0]+Mr[0,1]))

print("SPECİFİCİTY ORANI")
print(Mr[1,1]/(Mr[1,0]+Mr[1,1]))




































