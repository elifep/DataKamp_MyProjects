# 📘 IMDb Sentiment Analysis (Duygu Analizi Projesi)

## 🎯 Proje Açıklaması

Bu proje, IMDb film yorumları veri seti kullanılarak geliştirilmiş bir **Duygu Analizi (Sentiment Analysis)** modelidir. Projenin amacı, bir film yorumunun **olumlu (positive)** veya **olumsuz (negative)** olduğunu makine öğrenmesi yöntemleri ile sınıflandırmaktır.

## 📊 Veri Seti

- **Kaynak**: HuggingFace Datasets
- **Veri Seti**: IMDb Sentiment Dataset
- **Toplam Örnek**: 50.000 film yorumu
- **Etiketler**: Pozitif (1) / Negatif (0)
- **Eğitim/Test Ayrımı**: Veri seti zaten eğitim ve test olarak ayrılmıştır

## 🔧 Kullanılan Teknolojiler

- **Python 3.x**
- **scikit-learn**: Makine öğrenmesi modeli ve metrikler
- **datasets**: HuggingFace veri setlerini yükleme
- **nltk**: Doğal dil işleme (stopwords, lemmatization)
- **pandas**: Veri manipülasyonu
- **numpy**: Sayısal işlemler
- **matplotlib & seaborn**: Görselleştirme

## 📂 Proje Yapısı

```
project/
├── README.md
├── requirements.txt
├── sentiment_analysis.py
└── results/
    ├── metrics.txt
    └── confusion_matrix.png
```

## 🚀 Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Script'i çalıştırın:
```bash
python sentiment_analysis.py
```

## 📝 Metin Ön İşleme (Preprocessing) Adımları

Bu projede aşağıdaki preprocessing adımları uygulanmıştır:

### 1. **Küçük Harfe Çevirme (Lowercasing)**
   - Tüm metinler küçük harfe çevrilir
   - Örnek: "This Movie" → "this movie"

### 2. **Noktalama İşaretlerini Temizleme**
   - Noktalama işaretleri kaldırılır
   - Sadece harf ve boşluk karakterleri kalır
   - Örnek: "Hello, world!" → "Hello world"

### 3. **Sayıları Kaldırma**
   - Metinlerdeki sayılar kaldırılır
   - Noktalama temizleme sırasında otomatik olarak yapılır

### 4. **Stopwords Temizleme**
   - İngilizce stopwords (the, a, an, is, are, vb.) kaldırılır
   - Anlam taşımayan yaygın kelimeler filtrelenir
   - NLTK stopwords listesi kullanılır

### 5. **Lemmatization**
   - Kelimeler köklerine indirgenir
   - Örnek: "running" → "run", "better" → "good"
   - WordNetLemmatizer kullanılır

### 6. **Gereksiz Boşlukları Silme**
   - Birden fazla boşluk tek boşluğa indirgenir
   - Başta ve sonda boşluklar temizlenir

## 🔍 Özellik Çıkarımı: TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** kullanılarak metinler sayısal özelliklere dönüştürülmüştür.

### TF-IDF Parametreleri

1. **max_features=5000**
   - En yüksek TF-IDF skoruna sahip 5000 özellik seçilir
   - Boyut azaltma ve performans optimizasyonu için

2. **ngram_range=(1, 2)**
   - Unigram (tek kelime) ve bigram (iki kelime) kombinasyonları kullanılır
   - Örnek: "good movie" bigram olarak da özellik olarak eklenir

3. **min_df=2**
   - Bir kelime en az 2 dokümanda geçmelidir
   - Çok nadir kelimeleri filtreler

4. **max_df=0.95**
   - Bir kelime en fazla %95 dokümanda geçebilir
   - Çok yaygın kelimeleri (stopwords gibi) filtreler

5. **stop_words='english'**
   - İngilizce stopwords'leri otomatik olarak kaldırır
   - TF-IDF vektörleştiricinin kendi stopwords listesi kullanılır

## 🤖 Makine Öğrenmesi Modeli

### Seçilen Model: **Logistic Regression**

### Model Seçim Gerekçesi

Logistic Regression modeli seçilmiştir çünkü:

1. **Etkililik**: Metin sınıflandırma problemlerinde çok etkilidir
2. **Hız**: Eğitim ve tahmin süreleri kısadır
3. **Yorumlanabilirlik**: Model katsayıları özelliklerin önemini gösterir
4. **Düşük Overfitting Riski**: Düzenlileştirme (regularization) ile overfitting'i önler
5. **Olasılık Çıktısı**: Sadece sınıf değil, olasılık skorları da verir

### Model Parametreleri

- **max_iter=1000**: Maksimum iterasyon sayısı
- **random_state=42**: Tekrarlanabilirlik için
- **solver='lbfgs'**: Optimizasyon algoritması

## 📈 Model Performans Metrikleri

Model performansı aşağıdaki metriklerle değerlendirilmiştir:

- **Accuracy (Doğruluk)**: Genel doğru tahmin oranı
- **Precision (Kesinlik)**: Pozitif olarak tahmin edilenlerin gerçekten pozitif olma oranı
- **Recall (Duyarlılık)**: Gerçek pozitiflerin ne kadarının bulunduğu
- **F1-Score**: Precision ve Recall'un harmonik ortalaması

Detaylı metrikler `results/metrics.txt` dosyasında bulunmaktadır.

## 📊 Confusion Matrix

Model performansının görselleştirilmesi için confusion matrix oluşturulmuştur. Görsel `results/confusion_matrix.png` dosyasında bulunmaktadır.

## 🧪 Örnek Tahminler

Aşağıda modelin farklı cümleler üzerindeki tahminleri gösterilmiştir:

1. **"This movie was absolutely amazing! I loved every minute of it."**
   - Tahmin: **Positive**

2. **"This movie was boring and slow. I fell asleep halfway through."**
   - Tahmin: **Negative**

3. **"The acting was terrible and the plot made no sense."**
   - Tahmin: **Negative**

4. **"One of the best films I have ever seen. Highly recommended!"**
   - Tahmin: **Positive**

5. **"Waste of time. The story was confusing and the characters were flat."**
   - Tahmin: **Negative**

6. **"Brilliant cinematography and outstanding performances by all actors."**
   - Tahmin: **Positive**

7. **"I was disappointed with this film. It didn't live up to the hype."**
   - Tahmin: **Negative**

## 📁 Çıktı Dosyaları

### results/metrics.txt
- Model performans metrikleri
- TF-IDF parametreleri
- Örnek cümle tahminleri

### results/confusion_matrix.png
- Confusion matrix görselleştirmesi
- Gerçek ve tahmin edilen etiketlerin karşılaştırması

## 🔄 Çalıştırma

Projeyi çalıştırmak için:

```bash
# 1. Gerekli paketleri yükle
pip install -r requirements.txt

# 2. Script'i çalıştır
python sentiment_analysis.py
```

Script çalıştırıldığında:
1. IMDb veri seti yüklenir
2. Metinler ön işlenir
3. TF-IDF özellikleri çıkarılır
4. Model eğitilir
5. Model değerlendirilir
6. Sonuçlar kaydedilir

## 📌 Notlar

- İlk çalıştırmada NLTK verileri otomatik olarak indirilecektir
- Tüm veri setini kullanmak zaman alabilir, performans için örneklem alınabilir
- Model eğitimi tamamlandıktan sonra sonuçlar `results/` klasörüne kaydedilir

## 👤 Geliştirici

Bu proje DataKamp ödevi kapsamında geliştirilmiştir.

## 📄 Lisans

Bu proje eğitim amaçlıdır.

