# Deney 1 Adim Adim Islem Akisi

Bu not, kok dizindeki ilk pilot model klasorleri, `experiment_config.json`,
`test_metrics_summary.csv`, `test_per_image_metrics.csv`, `run_summary.json`
dosyalari ve `analysis_review/experiment_1_2_fairness_notes.md` okunarak
hazirlanmistir. Amac, Deney 1'de model taramasinin hangi sirayla yapildigini
ve sonuclarin nasil okunmasi gerektigini kaydetmektir.

## 1. Deneyin temel amaci

Deney 1, ReCodAI-LUC veri kumesi uzerinde ilk genis model taramasidir.
Bu asamada hedef final sistemi kurmak degil, cok sayida aday mimarinin
kucuk ve dengeli bir pilot subset uzerindeki davranisini hizli ve adil
sekilde gormektir.

Deney 1 kapsaminda 15 aday model klasoru uretildi:

```text
plain_unet
unetplusplus
efficientnetb0_unet
resnet50_unet
deeplabv3plus
segformer_b0
dinov2_seg
cmfdformer
busternet
doagan
mantranet
mvssnet
mvssnetpp
qdl_cmfd
selfcorr_cmfd
siamese_cmfd
```

Bu nedenle Deney 1'in rolu:

```text
genis aday havuzu -> ayni pilot subset -> ayni split/protokol -> ilk model siralamasi
```

## 2. Veri subset'i ve split

Deney 1 tam veri kumesini kullanmadi. Ilk pilot icin sinif basina 300 ornek
alindi:

| Sinif | Ornek sayisi |
|---|---:|
| Authentic | 300 |
| Forged | 300 |
| Toplam | 600 |

Split ayarlari:

| Ayar | Deger |
|---|---|
| Seed | 42 |
| Split orani | 60 / 20 / 20 |
| Train | yaklasik 360 goruntu |
| Validation / threshold tuning | yaklasik 120 goruntu |
| Internal test | yaklasik 120 goruntu |
| Split tipi | stratified, group-aware |
| Grup anahtari | `image_id` / `group_id` |

`image_id` grup anahtari kullanildigi icin ayni goruntu grubunun train,
validation ve internal-test arasinda karismamasi hedeflendi.

## 3. Girdi ve maske hazirlama

Her model icin temel veri temsili ayni tutuldu:

```text
PNG image oku
-> image'i 256x256 boyutuna resize et
-> forged ise .npy maskeyi oku ve binary maskeye cevir
-> authentic ise tamamen sifir maske uret
-> train tarafinda augmentasyon uygula
-> validation/test tarafinda sadece deterministik on isleme uygula
```

Forged maskelerde pozitif piksel `mask > 0` mantigiyla alindi. Authentic
goruntulerde ground-truth maske tamamen sifirdir. Bu nedenle all-image pixel
metrikleri tek basina fazla iyimser okunabilir; forged-only lokalizasyon
metrikleri daha anlamlidir.

## 4. Ortak egitim ayarlari

Kok model config dosyalarinda ortak ayarlar su sekildedir:

| Ayar | Deger |
|---|---|
| Image size | 256x256 |
| Batch size | 8 |
| Epoch ust siniri | 40 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Early stopping patience | 8 |
| LR scheduler | ReduceLROnPlateau |
| Image decision mode | `max_probability` |
| Threshold adaylari | 0.10 - 0.90, 0.05 adim |

Train tarafinda model agirliklari egitildi. Validation tarafinda threshold
secildi. Internal-test tarafinda secilen threshold sabit uygulanarak final
pilot sonucu raporlandi.

## 5. Threshold secimi

Deney 1'de pixel threshold test setinden secilmedi.

Islem sirasi:

```text
validation probability map uret
-> threshold adaylarini tara
-> validation metrigine gore en iyi threshold'u sec
-> internal-test probability map uret
-> secilen threshold'u internal-test'e sabit uygula
```

Threshold adaylari:

```text
0.10, 0.15, 0.20, ..., 0.90
```

Image-level karar icin probability map'ten tek skor uretildi. Ana skor
`max_probability` idi:

```text
image_score = probability_map icindeki maksimum olasilik
```

Bu skor, goruntude herhangi bir yerde sahtecilik sinyali olup olmadigini
yakalamak icin kullanildi.

## 6. Hesaplanan metrikler

Her modelde temel ciktilar:

```text
threshold_analysis.csv
test_per_image_metrics.csv
test_metrics_summary.csv
test_confusion_matrix.csv
test_roc_auc_data.csv
test_precision_recall_data.csv
prediction_examples/
```

Pixel-level metrikler:

| Metrik | Anlami |
|---|---|
| Pixel F1 / Dice | Tahmin maskesi ile GT maskesi arasindaki overlap dengesi |
| Pixel IoU | Kesisim / birlesim |
| Precision | Tahmin edilen pozitif piksellerin ne kadari dogru |
| Recall | GT pozitif piksellerin ne kadari yakalandi |
| Specificity | Negatif piksellerin ne kadari negatif kaldi |

Image-level metrikler:

| Metrik | Anlami |
|---|---|
| Image F1 | authentic/forged goruntu karari F1 |
| ROC-AUC | image score'un sinif ayirma gucu |
| Sensitivity | forged goruntuyu forged deme orani |
| Specificity | authentic goruntuyu authentic tutma orani |

## 7. Sonuc okuma mantigi

Deney 1 genis bir tarama oldugu icin tek bir metrikle karar verilmedi.
Ozellikle su ayrim korundu:

```text
all-image pixel metric: authentic sifir maskelerden etkilenebilir
forged-only/localization metric: sahtecilik varken lokalizasyon basarisi
image-level metric: goruntu forged mi authentic mi karari
```

Kok model klasorlerindeki `test_metrics_summary.csv` dosyalari farkli model
ailesi ve eski kosu formatlari nedeniyle tamamen ayni semantik ayrintiyi
tasmayabilir. Bu yuzden Deney 1, daha cok aday eleme ve sonraki deneylere
model secme amaciyla kullanildi.

## 8. Deney 2'ye gecis gerekcesi

Deney 1'de tek seed ve tek pilot split uzerinde bazi modeller guclu gorundu.
Fakat tek kosu ile modelin gercekten kararli olup olmadigi anlasilamaz.

Bu nedenle Deney 2'de Deney 1'in tamami tekrar edilmedi. Deney 1'den secilen
kisa liste, ayni 300 authentic + 300 forged goruntu havuzu uzerinde farkli
seed'lerle tekrar calistirildi.

Deney 1'den Deney 2'ye tasinan ana soru:

```text
Bu modellerin siralamasi sadece seed/split sansi mi,
yoksa kararli bir performans sinyali mi?
```

## 9. Kaynak dosyalar

Ana kaynaklar:

```text
{model}/experiment_config.json
{model}/threshold_analysis.csv
{model}/test_per_image_metrics.csv
{model}/test_metrics_summary.csv
{model}/run_summary.json
analysis_review/experiment_1_2_fairness_notes.md
```

Model klasorleri:

```text
plain_unet
unetplusplus
efficientnetb0_unet
resnet50_unet
deeplabv3plus
segformer_b0
dinov2_seg
cmfdformer
busternet
doagan
mantranet
mvssnet
mvssnetpp
qdl_cmfd
selfcorr_cmfd
siamese_cmfd
```
