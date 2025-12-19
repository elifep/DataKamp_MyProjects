"""
IMDb Sentiment Analysis - Klasik Makine Öğrenmesi (ML)
Bu script TF-IDF + Logistic Regression kullanarak sentiment analysis yapar.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')

# NLTK verilerini indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Sonuçlar klasörünü oluştur
os.makedirs('results/confusion_matrices', exist_ok=True)

print("=" * 60)
print("IMDb Sentiment Analysis - Klasik ML (TF-IDF + Logistic Regression)")
print("=" * 60)

# Eğitim süresini ölç
start_time = time.time()

# ==================== 1. VERİ SETİNİ YÜKLEME ====================
print("\n[1/5] Veri seti yükleniyor...")
dataset = load_dataset("imdb")

train_data = dataset['train']
test_data = dataset['test']

train_df = pd.DataFrame({
    'text': train_data['text'],
    'label': train_data['label']
})

test_df = pd.DataFrame({
    'text': test_data['text'],
    'label': test_data['label']
})

print(f"Eğitim verisi: {len(train_df)} örnek")
print(f"Test verisi: {len(test_df)} örnek")

# ==================== 2. METİN ÖN İŞLEME ====================
print("\n[2/5] Metin ön işleme yapılıyor...")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Metin ön işleme: küçük harf, noktalama temizleme, stopwords, lemmatization"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Eğitim verisi ön işleniyor...")
train_df['processed_text'] = train_df['text'].apply(preprocess_text)

print("Test verisi ön işleniyor...")
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

print("Ön işleme tamamlandı!")

# ==================== 3. ÖZELLİK ÇIKARIMI (TF-IDF) ====================
print("\n[3/5] TF-IDF ile özellik çıkarımı yapılıyor...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.95
)

print("TF-IDF vektörleri oluşturuluyor...")
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['processed_text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['processed_text'])

y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"Eğitim vektör boyutu: {X_train_tfidf.shape}")
print(f"Test vektör boyutu: {X_test_tfidf.shape}")

# ==================== 4. MODEL EĞİTİMİ ====================
print("\n[4/5] Model eğitiliyor...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

train_start = time.time()
print("Logistic Regression modeli eğitiliyor...")
model.fit(X_train_tfidf, y_train)
train_time = time.time() - train_start
print(f"Model eğitimi tamamlandı! Süre: {train_time:.2f} saniye")

# ==================== 5. MODEL DEĞERLENDİRME ====================
print("\n[5/5] Model değerlendiriliyor...")

y_pred = model.predict(X_test_tfidf)

# Metrikleri hesapla
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

total_time = time.time() - start_time

print("\n" + "=" * 60)
print("MODEL PERFORMANS METRİKLERİ")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"\nEğitim Süresi: {train_time:.2f} saniye")
print(f"Toplam Süre: {total_time:.2f} saniye")
print("=" * 60)

# Metrikleri dosyaya kaydet
with open('results/ml_metrics.txt', 'w', encoding='utf-8') as f:
    f.write("IMDb Sentiment Analysis - Klasik ML Model Performans Metrikleri\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model: Logistic Regression\n")
    f.write("Özellik Çıkarımı: TF-IDF\n\n")
    f.write("METRİKLER:\n")
    f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n\n")
    f.write("SÜRE:\n")
    f.write(f"Eğitim Süresi: {train_time:.2f} saniye\n")
    f.write(f"Toplam Süre: {total_time:.2f} saniye\n\n")
    f.write("TF-IDF Parametreleri:\n")
    f.write(f"  - max_features: 5000\n")
    f.write(f"  - ngram_range: (1, 2)\n")
    f.write(f"  - min_df: 2\n")
    f.write(f"  - max_df: 0.95\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Klasik ML (TF-IDF + Logistic Regression)', fontsize=16, pad=20)
plt.ylabel('Gerçek Etiket', fontsize=12)
plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrices/ml_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nConfusion matrix kaydedildi: results/confusion_matrices/ml_confusion_matrix.png")

print("\n" + "=" * 60)
print("KLASİK ML MODELİ TAMAMLANDI!")
print("=" * 60)


