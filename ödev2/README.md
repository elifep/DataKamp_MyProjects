# IMDb Sentiment Analysis - Derin Öğrenme (RNN/LSTM)

## Proje Açıklaması

Bu proje, IMDb film yorumları veri seti kullanılarak **Klasik Makine Öğrenmesi** ve **Derin Öğrenme** yaklaşımlarını karşılaştırmaktadır. İki farklı model geliştirilmiştir:

1. **Klasik ML Modeli**: TF-IDF + Logistic Regression
2. **Derin Öğrenme Modeli**: LSTM (Long Short-Term Memory)

## Proje Yapısı

```
ödev2/
├── README.md
├── requirements.txt
├── sentiment_analysis_ml.py      # Klasik ML modeli
├── sentiment_analysis_rnn.py      # LSTM modeli
└── results/
    ├── ml_metrics.txt
    ├── rnn_metrics.txt
    ├── rnn_training_history.png
    └── confusion_matrices/
        ├── ml_confusion_matrix.png
        └── rnn_confusion_matrix.png
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Klasik ML Modelini Çalıştırma
```bash
python sentiment_analysis_ml.py
```

### LSTM Modelini Çalıştırma
```bash
python sentiment_analysis_rnn.py
```

## Model Detayları

### 1. Klasik ML Modeli (TF-IDF + Logistic Regression)

#### Özellik Çıkarımı: TF-IDF
- **max_features**: 5000
- **ngram_range**: (1, 2) - Unigram ve bigram
- **min_df**: 2
- **max_df**: 0.95

#### Model: Logistic Regression
- Hızlı eğitim süresi
- Yorumlanabilir katsayılar
- Düşük bellek kullanımı

#### Metin Ön İşleme:
- Küçük harfe çevirme
- Noktalama temizleme
- Stopwords kaldırma
- Lemmatization

### 2. LSTM Modeli

#### Model Mimarisi:
```
Embedding Layer (vocab_size=10000, embedding_dim=128)
    ↓
LSTM Layer (128 units, dropout=0.2)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (1 unit, Sigmoid)
```

#### Parametreler:
- **vocab_size**: 10000 - Kelime dağarcığı boyutu
- **max_length**: 200 - Maksimum cümle uzunluğu
- **embedding_dim**: 128 - Embedding vektör boyutu
- **LSTM units**: 128
- **Batch size**: 64
- **Epochs**: 10 (Early stopping ile)

#### Metin Ön İşleme:
- Küçük harfe çevirme
- Noktalama temizleme
- **Not**: Stopwords kaldırılmıyor (LSTM için kelime sırası önemli)

#### Embedding Layer Açıklaması:
- **vocab_size**: En sık kullanılan 10000 kelimeyi içerir
- **embedding_dim**: Her kelime 128 boyutlu bir vektöre dönüştürülür
- **max_length**: Her cümle maksimum 200 kelime uzunluğunda olacak şekilde padding/truncation yapılır

## Karşılaştırma Analizi

### 1. Performans Karşılaştırması

| Metrik | TF-IDF + ML | LSTM |
|--------|-------------|------|
| **Accuracy** | ~87.5% | ~85-88% |
| **Precision** | ~87.2% | ~85-87% |
| **Recall** | ~88.0% | ~85-88% |
| **F1-Score** | ~87.6% | ~85-87% |

**Sonuç**: Her iki model de benzer performans gösterir. Klasik ML modeli genellikle biraz daha yüksek accuracy elde eder.

### 2. Eğitim Süresi

| Model | Eğitim Süresi |
|-------|---------------|
| **TF-IDF + ML** | ~10-30 saniye |
| **LSTM** | ~5-15 dakika (GPU ile daha hızlı) |

**Sonuç**: Klasik ML modeli çok daha hızlı eğitilir. LSTM modeli daha uzun süre gerektirir.

### 3. Overfitting Eğilimi

#### TF-IDF + ML:
- Düşük overfitting riski
- Düzenlileştirme (regularization) ile kontrol edilir
- Validation set'te tutarlı performans

#### LSTM:
- Yüksek overfitting riski
- Dropout ve Early Stopping kullanılır
- Eğitim geçmişi grafiği ile izlenir
- Validation loss artışı durumunda eğitim durdurulur

**Sonuç**: LSTM modeli daha fazla overfitting eğilimi gösterir, bu yüzden dikkatli eğitim gerekir.

### 4. Yorumlanabilirlik

#### TF-IDF + ML:
- ✅ Yüksek yorumlanabilirlik
- Katsayılar özelliklerin önemini gösterir
- Hangi kelimelerin pozitif/negatif olduğu görülebilir
- Model kararları açıklanabilir

#### LSTM:
- ❌ Düşük yorumlanabilirlik
- "Kara kutu" modeli
- İç işleyişi anlaşılması zor
- Hangi özelliklerin kullanıldığı belirsiz

**Sonuç**: Klasik ML modeli çok daha yorumlanabilirdir.

### 5. Bellek ve Kaynak Kullanımı

| Özellik | TF-IDF + ML | LSTM |
|---------|-------------|------|
| **Bellek Kullanımı** | Düşük | Yüksek |
| **CPU/GPU** | CPU yeterli | GPU önerilir |
| **Model Boyutu** | Küçük (~MB) | Büyük (~100MB+) |

### 6. Veri Gereksinimleri

- **TF-IDF + ML**: Küçük veri setlerinde de iyi çalışır
- **LSTM**: Büyük veri setlerinde daha iyi performans gösterir

## Avantajlar ve Dezavantajlar

### Klasik ML (TF-IDF + Logistic Regression)

**Avantajlar:**
- ✅ Hızlı eğitim
- ✅ Yorumlanabilir
- ✅ Düşük kaynak gereksinimi
- ✅ Küçük veri setlerinde etkili
- ✅ Overfitting riski düşük

**Dezavantajlar:**
- ❌ Kelime sırasını dikkate almaz
- ❌ Uzun bağımlılıkları yakalayamaz
- ❌ Bağlam anlayışı sınırlı

### LSTM

**Avantajlar:**
- ✅ Kelime sırasını dikkate alır
- ✅ Uzun bağımlılıkları yakalayabilir
- ✅ Bağlam anlayışı daha iyi
- ✅ Büyük veri setlerinde güçlü

**Dezavantajlar:**
- ❌ Yavaş eğitim
- ❌ Yorumlanabilirlik düşük
- ❌ Yüksek kaynak gereksinimi
- ❌ Overfitting riski yüksek

## Sonuç ve Öneriler

### Hangi Modeli Seçmeli?

**Klasik ML Modeli Seçin Eğer:**
- Hızlı sonuç gerekiyorsa
- Yorumlanabilirlik önemliyse
- Sınırlı kaynaklarınız varsa
- Küçük-orta boyutlu veri setiniz varsa

**LSTM Modeli Seçin Eğer:**
- Kelime sırası ve bağlam çok önemliyse
- Büyük veri setiniz varsa
- GPU erişiminiz varsa
- En yüksek performansı hedefliyorsanız

### Genel Değerlendirme

Bu proje için **Klasik ML modeli** daha pratik bir seçimdir çünkü:
1. Benzer performans gösterir
2. Çok daha hızlıdır
3. Yorumlanabilirdir
4. Daha az kaynak gerektirir

Ancak, **LSTM modeli** daha karmaşık metinlerde ve büyük veri setlerinde avantaj sağlayabilir.

## Teknik Detaylar

### Tokenization ve Padding

LSTM modelinde:
- Metinler tokenize edilir (kelimeler sayılara dönüştürülür)
- Tüm cümleler aynı uzunluğa getirilir (padding)
- Uzun cümleler kesilir (truncation)
- Kısa cümleler sıfırlarla doldurulur

### Embedding Layer

- Her kelime, anlamını temsil eden bir vektöre dönüştürülür
- Benzer anlamlı kelimeler benzer vektörlere sahiptir
- Model eğitimi sırasında öğrenilir

### Early Stopping

- Validation loss artmaya başladığında eğitim durdurulur
- Overfitting'i önlemek için kullanılır
- En iyi model ağırlıkları geri yüklenir

## Geliştirici Notları

- Her iki model de aynı veri setini kullanır
- Metrikler `results/` klasörüne kaydedilir
- Confusion matrix'ler görselleştirilir
- LSTM modeli için eğitim geçmişi grafiği oluşturulur

## Lisans

Bu proje eğitim amaçlıdır.


