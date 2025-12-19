"""
IMDb Sentiment Analysis - Derin Öğrenme (RNN/LSTM)
Bu script LSTM kullanarak sentiment analysis yapar.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import re
import warnings
warnings.filterwarnings('ignore')

# Sonuçlar klasörünü oluştur
os.makedirs('results/confusion_matrices', exist_ok=True)

print("=" * 60)
print("IMDb Sentiment Analysis - Derin Öğrenme (LSTM)")
print("=" * 60)

# Eğitim süresini ölç
start_time = time.time()

# ==================== 1. VERİ SETİNİ YÜKLEME ====================
print("\n[1/6] Veri seti yükleniyor...")
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
print("\n[2/6] Metin ön işleme yapılıyor...")

def preprocess_text(text):
    """
    Metin ön işleme:
    1. Küçük harfe çevirme
    2. Noktalama işaretlerini temizleme
    3. Gereksiz boşlukları silme
    Not: Stopwords kaldırılmıyor (LSTM için kelime sırası önemli)
    """
    # Küçük harfe çevirme
    text = text.lower()
    
    # Noktalama işaretlerini temizleme (sadece harf, rakam ve boşluk bırak)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Gereksiz boşlukları silme
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("Eğitim verisi ön işleniyor...")
train_df['processed_text'] = train_df['text'].apply(preprocess_text)

print("Test verisi ön işleniyor...")
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

print("Ön işleme tamamlandı!")

# ==================== 3. TOKENIZATION VE PADDING ====================
print("\n[3/6] Tokenization ve padding yapılıyor...")

# Tokenizer parametreleri
vocab_size = 10000  # En sık kullanılan 10000 kelime
max_length = 200    # Maksimum cümle uzunluğu
oov_tok = "<OOV>"   # Out-of-vocabulary token

# Tokenizer oluştur
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_df['processed_text'])

# Metinleri sequence'lere çevir
X_train_seq = tokenizer.texts_to_sequences(train_df['processed_text'])
X_test_seq = tokenizer.texts_to_sequences(test_df['processed_text'])

# Padding/Truncation uygula
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"Eğitim verisi boyutu: {X_train_padded.shape}")
print(f"Test verisi boyutu: {X_test_padded.shape}")
print(f"Kelime dağarcığı boyutu: {vocab_size}")

# ==================== 4. MODEL MİMARİSİ ====================
print("\n[4/6] LSTM model mimarisi oluşturuluyor...")

# Embedding parametreleri
embedding_dim = 128  # Embedding vektör boyutu

model = Sequential([
    # Embedding Layer
    Embedding(input_dim=vocab_size,      # vocab_size: Kelime dağarcığı boyutu
              output_dim=embedding_dim,   # embedding_dim: Her kelime için vektör boyutu
              input_length=max_length,    # max_length: Maksimum cümle uzunluğu
              name='embedding'),
    
    # LSTM Layer
    LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False, name='lstm'),
    
    # Dense Layer
    Dense(64, activation='relu', name='dense1'),
    Dropout(0.5, name='dropout'),
    
    # Output Layer
    Dense(1, activation='sigmoid', name='output')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Özeti:")
model.summary()

# ==================== 5. MODEL EĞİTİMİ ====================
print("\n[5/6] Model eğitiliyor...")

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Eğitim parametreleri
batch_size = 64
epochs = 10

train_start = time.time()

history = model.fit(
    X_train_padded, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

train_time = time.time() - train_start
print(f"\nModel eğitimi tamamlandı! Süre: {train_time:.2f} saniye")

# ==================== 6. MODEL DEĞERLENDİRME ====================
print("\n[6/6] Model değerlendiriliyor...")

# Tahminler
y_pred_proba = model.predict(X_test_padded, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

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
with open('results/rnn_metrics.txt', 'w', encoding='utf-8') as f:
    f.write("IMDb Sentiment Analysis - LSTM Model Performans Metrikleri\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model: LSTM (Long Short-Term Memory)\n")
    f.write("Özellik Çıkarımı: Embedding Layer\n\n")
    f.write("METRİKLER:\n")
    f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n\n")
    f.write("SÜRE:\n")
    f.write(f"Eğitim Süresi: {train_time:.2f} saniye\n")
    f.write(f"Toplam Süre: {total_time:.2f} saniye\n\n")
    f.write("MODEL PARAMETRELERİ:\n")
    f.write(f"  - vocab_size: {vocab_size}\n")
    f.write(f"  - max_length: {max_length}\n")
    f.write(f"  - embedding_dim: {embedding_dim}\n")
    f.write(f"  - LSTM units: 128\n")
    f.write(f"  - Batch size: {batch_size}\n")
    f.write(f"  - Epochs: {epochs}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - LSTM Model', fontsize=16, pad=20)
plt.ylabel('Gerçek Etiket', fontsize=12)
plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrices/rnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nConfusion matrix kaydedildi: results/confusion_matrices/rnn_confusion_matrix.png")

# Eğitim geçmişi görselleştirme
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
plt.plot(history.history['val_accuracy'], label='Doğrulama Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Doğrulama Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/rnn_training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("Eğitim geçmişi kaydedildi: results/rnn_training_history.png")

print("\n" + "=" * 60)
print("LSTM MODELİ TAMAMLANDI!")
print("=" * 60)


