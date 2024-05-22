import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report, roc_curve, auc
import ssl

ssl._create_default_https_context = ssl._create_unverified_context # SSL sertifikası hatası aldım ve bu kodla sorunu çözdüm

# Veri setini yükle
online_retail = pd.read_excel('online-retail.xlsx')

# Eksik değerleri kontrol et
print(online_retail.isnull().sum())

# Eksik değerleri olan satırları düşür
online_retail.dropna(inplace=True)

# InvoiceDate kolonunu datetime tipine dönüştür
online_retail['InvoiceDate'] = pd.to_datetime(online_retail['InvoiceDate'])

# Ay, gün ve saat kolonlarını oluştur
online_retail['Month'] = online_retail['InvoiceDate'].dt.month
online_retail['Day'] = online_retail['InvoiceDate'].dt.day
online_retail['Hour'] = online_retail['InvoiceDate'].dt.hour

# Quantity kolonunu hedef değişken olarak kullanarak pozitif ve negatif olarak etiketle
online_retail['QuantityLabel'] = np.where(online_retail['Quantity'] > 0, 1, 0)

# Kategorik kolonları one-hot encoding ile dönüştür
online_retail = pd.get_dummies(online_retail, columns=['Country'])

# İlk birkaç satırı tekrar görüntüle
print(online_retail.head())

# Özellikler ve hedef değişkeni ayır
X = online_retail[['UnitPrice', 'Month', 'Day', 'Hour'] + [col for col in online_retail.columns if 'Country_' in col]]
y = online_retail['QuantityLabel']

# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi standardize et
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree modelini oluştur ve eğit
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Test seti üzerinde tahminler yap
y_pred = dt_model.predict(X_test_scaled)

# Performans metriklerini hesapla
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred)
specificity = recall_score(y_test, y_pred, pos_label=0)
precision = precision_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Sonuçları yazdır
print(f'Accuracy: {accuracy}')
print(f'Sensitivity: {sensitivity}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'Specificity: {specificity}')
print(f'F1 Score: {f1_score}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix'i görselleştir
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC eğrisini hesapla ve görselleştir
y_pred_proba = dt_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()