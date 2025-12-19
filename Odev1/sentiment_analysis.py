"""
IMDb Sentiment Analysis Projesi
Bu script IMDb film yorumlarını pozitif/negatif olarak sınıflandırır.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
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
os.makedirs('results', exist_ok=True)

print("=" * 60)
print("IMDb Sentiment Analysis Projesi")
print("=" * 60)

# ==================== 1. VERİ SETİNİ YÜKLEME ====================
print("\n[1/6] Veri seti yükleniyor...")
dataset = load_dataset("imdb")

# Eğitim ve test verilerini al
train_data = dataset['train']
test_data = dataset['test']

# Veriyi DataFrame'e çevir
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
print(f"Pozitif örnekler (eğitim): {train_df['label'].sum()}")
print(f"Negatif örnekler (eğitim): {len(train_df) - train_df['label'].sum()}")

# Performans için eğitim verisinden örneklem al (isteğe bağlı - tüm veriyi kullanmak için yorum satırını kaldırın)
# train_df = train_df.sample(n=10000, random_state=42)  # Hızlı test için

# ==================== 2. METİN ÖN İŞLEME ====================
print("\n[2/6] Metin ön işleme yapılıyor...")

# Stopwords ve lemmatizer'ı yükle
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Metin ön işleme fonksiyonu:
    1. Küçük harfe çevirme
    2. Noktalama işaretlerini temizleme
    3. Sayıları kaldırma
    4. Stopwords temizleme
    5. Lemmatization
    6. Gereksiz boşlukları silme
    """
    # 1. Küçük harfe çevirme
    text = text.lower()
    
    # 2. Noktalama işaretlerini temizleme (sadece harf ve boşluk bırak)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Sayıları kaldırma (zaten yukarıdaki regex ile kaldırıldı)
    
    # 4. Stopwords temizleme ve lemmatization
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # 5. Gereksiz boşlukları silme
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Ön işleme uygula
print("Eğitim verisi ön işleniyor...")
train_df['processed_text'] = train_df['text'].apply(preprocess_text)

print("Test verisi ön işleniyor...")
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

print("Ön işleme tamamlandı!")
print(f"Örnek işlenmiş metin: {train_df['processed_text'].iloc[0][:100]}...")

# ==================== 3. ÖZELLİK ÇIKARIMI (TF-IDF) ====================
print("\n[3/6] TF-IDF ile özellik çıkarımı yapılıyor...")

# TF-IDF vektörleştirici oluştur
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,        # En önemli 5000 özellik
    ngram_range=(1, 2),      # Unigram ve bigram kullan
    stop_words='english',    # İngilizce stopwords'leri kaldır
    min_df=2,                 # En az 2 dokümanda geçmeli
    max_df=0.95              # En fazla %95 dokümanda geçmeli
)

# TF-IDF vektörlerini oluştur
print("TF-IDF vektörleri oluşturuluyor...")
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['processed_text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['processed_text'])

y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"Eğitim vektör boyutu: {X_train_tfidf.shape}")
print(f"Test vektör boyutu: {X_test_tfidf.shape}")

# ==================== 4. MAKİNE ÖĞRENMESİ MODELİ EĞİTİMİ ====================
print("\n[4/6] Model eğitiliyor...")

# Logistic Regression modeli seçildi
# Sebep: Metin sınıflandırma için etkili, hızlı ve yorumlanabilir
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

print("Logistic Regression modeli eğitiliyor...")
model.fit(X_train_tfidf, y_train)
print("Model eğitimi tamamlandı!")

# ==================== 5. MODEL DEĞERLENDİRME ====================
print("\n[5/6] Model değerlendiriliyor...")

# Tahminler
y_pred = model.predict(X_test_tfidf)

# Metrikleri hesapla
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANS METRİKLERİ")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("=" * 60)

# Metrikleri dosyaya kaydet
with open('results/metrics.txt', 'w', encoding='utf-8') as f:
    f.write("IMDb Sentiment Analysis - Model Performans Metrikleri\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("Model: Logistic Regression\n")
    f.write("Özellik Çıkarımı: TF-IDF\n")
    f.write(f"TF-IDF Parametreleri:\n")
    f.write(f"  - max_features: 5000\n")
    f.write(f"  - ngram_range: (1, 2)\n")
    f.write(f"  - min_df: 2\n")
    f.write(f"  - max_df: 0.95\n")

# Confusion Matrix oluştur
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - IMDb Sentiment Analysis', fontsize=16, pad=20)
plt.ylabel('Gerçek Etiket', fontsize=12)
plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix kaydedildi: results/confusion_matrix.png")

# ==================== 6. KENDİ CÜMLELERİYLE TEST ====================
print("\n[6/6] Örnek cümleler test ediliyor...")

# Test cümleleri
test_sentences = [
    "This movie was absolutely amazing! I loved every minute of it.",
    "This movie was boring and slow. I fell asleep halfway through.",
    "The acting was terrible and the plot made no sense.",
    "One of the best films I have ever seen. Highly recommended!",
    "Waste of time. The story was confusing and the characters were flat.",
    "Brilliant cinematography and outstanding performances by all actors.",
    "I was disappointed with this film. It didn't live up to the hype."
]

print("\n" + "=" * 60)
print("ÖRNEK CÜMLE TAHMİNLERİ")
print("=" * 60)

predictions_text = []

for sentence in test_sentences:
    # Ön işleme
    processed = preprocess_text(sentence)
    
    # TF-IDF dönüşümü
    sentence_tfidf = tfidf_vectorizer.transform([processed])
    
    # Tahmin
    prediction = model.predict(sentence_tfidf)[0]
    probability = model.predict_proba(sentence_tfidf)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[prediction] * 100
    
    print(f"\nCümle: \"{sentence}\"")
    print(f"Tahmin: {sentiment} (Güven: {confidence:.2f}%)")
    
    predictions_text.append({
        'cümle': sentence,
        'tahmin': sentiment,
        'güven': f"{confidence:.2f}%"
    })

# Tahminleri dosyaya kaydet
with open('results/metrics.txt', 'a', encoding='utf-8') as f:
    f.write("\n\n" + "=" * 60 + "\n")
    f.write("ÖRNEK CÜMLE TAHMİNLERİ\n")
    f.write("=" * 60 + "\n\n")
    for pred in predictions_text:
        f.write(f"Cümle: {pred['cümle']}\n")
        f.write(f"Tahmin: {pred['tahmin']} (Güven: {pred['güven']})\n\n")

print("\n" + "=" * 60)
print("TÜM İŞLEMLER TAMAMLANDI!")
print("=" * 60)
print("\nOluşturulan dosyalar:")
print("  - results/metrics.txt")
print("  - results/confusion_matrix.png")

# ==================== İNTERAKTİF CÜMLE TESTİ ====================
print("\n" + "=" * 60)
print("İNTERAKTİF CÜMLE TESTİ")
print("=" * 60)
print("Kendi cümlelerinizi test edebilirsiniz.")
print("Çıkmak için 'q' veya 'quit' yazın.\n")

while True:
    try:
        user_input = input("Test etmek istediğiniz cümleyi girin: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit', 'çık']:
            print("\nTest modundan çıkılıyor...")
            break
        
        if not user_input:
            print("Lütfen bir cümle girin!\n")
            continue
        
        # Ön işleme
        processed = preprocess_text(user_input)
        
        # TF-IDF dönüşümü
        sentence_tfidf = tfidf_vectorizer.transform([processed])
        
        # Tahmin
        prediction = model.predict(sentence_tfidf)[0]
        probability = model.predict_proba(sentence_tfidf)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probability[prediction] * 100
        
        print(f"\n{'='*60}")
        print(f"Cümle: \"{user_input}\"")
        print(f"Tahmin: {sentiment}")
        print(f"Güven: {confidence:.2f}%")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\nTest modundan çıkılıyor...")
        break
    except Exception as e:
        print(f"\nHata oluştu: {e}\n")
        continue

print("\nProgram sonlandırıldı.")

