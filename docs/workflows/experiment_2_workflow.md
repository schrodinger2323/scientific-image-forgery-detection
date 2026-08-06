# Deney 2 Adim Adim Islem Akisi

Bu not, `deney_2/pilot_experiment_design.md`,
`deney_2/pilot_experiment_interpretation.md`,
`deney_2/pilot_seed_comparison.csv`,
`deney_2/pilot_seed_per_run_results.csv` ve
`analysis_review/experiment_1_2_fairness_notes.md` dosyalari okunarak
hazirlanmistir.

## 1. Deneyin temel amaci

Deney 2, Deney 1'in devaminda yapilan seed stability deneyidir. Amac, Deney
1'de tek seed ile elde edilen model siralamasinin rastgele split, baslangic
agirligi veya augmentasyon sansindan kaynaklanip kaynaklanmadigini kontrol
etmektir.

Deney 2 su soruya cevap verir:

```text
Ayni goruntu havuzu korunursa, sadece seed degistiginde model performansi
ve model siralamasi ne kadar kararli kaliyor?
```

## 2. Deney 1'den farki

Deney 1:

```text
15 aday model
-> tek pilot subset
-> seed 42
-> genis model taramasi
```

Deney 2:

```text
5 secilmis model
-> ayni 300 authentic + 300 forged goruntu havuzu
-> seed 42, 123, 2025
-> seed kararliligi olcumu
```

Yani Deney 2, Deney 1'in 15 modelinin tamamini yeniden kosmadi. Yalnizca
secilen kisa listeyi farkli seed'lerle test etti.

## 3. Sabit tutulanlar

Deney 2'de adil karsilastirma icin su faktorler sabit tutuldu:

| Sabit faktor | Deger |
|---|---|
| Goruntu havuzu | Deney 1 ile ayni 300 authentic + 300 forged |
| Image size | 256x256 |
| Batch size | 8 |
| Epoch ust siniri | 40 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Split orani | 60 / 20 / 20 |
| Split tipi | stratified group-aware |
| Grup anahtari | `image_id` / `group_id` |
| Threshold tuning | sadece val-tune |
| Test kullanimi | sadece final internal-test evaluation |

Subset sabitleme su dosya mantigiyla yapildi:

```text
fixed_subset_index = _shared_splits_seed42_subset/full_dataset_index.csv
```

Bu sayede Deney 2'nin seedleri ayni sample_id havuzundan basladi.

## 4. Degistirilenler

Deney 2'de bilincli olarak degistirilen tek ana faktor seed'dir.

Seed degisince su seyler degisebilir:

```text
train / val-tune / internal-test atamasi
model baslangic agirliklari
dataloader shuffling
train-only augmentasyon rastgeleligi
```

Bu nedenle Deney 2 bir yeni model arama deneyi degildir; bir kararlilik
deneyidir.

## 5. Calistirilan modeller

Deney 2'de 5 model calistirildi:

```text
plain_unet
unetplusplus
efficientnetb0_unet
deeplabv3plus
segformer_b0
```

Seedler:

```text
42, 123, 2025
```

Her seed/model kombinasyonu icin ayri klasorde egitim loglari, threshold
analizi, test metrikleri ve prediction example dosyalari kaydedildi.

## 6. Veri ve maske akisi

Islem sirasi:

```text
fixed subset CSV oku
-> seed'e gore stratified group-aware split olustur
-> authentic icin sifir mask uret
-> forged icin .npy mask oku ve binary maskeye cevir
-> image/mask 256x256 resize
-> train augmentasyonu uygula
-> validation ve internal-test deterministik kalsin
```

Group-aware split sayesinde ayni `image_id` grubu farkli splitlere
dagitilmadi.

## 7. Egitim ve checkpoint secimi

Her model ayni temel egitim protokoluyle calistirildi:

```text
model egit
-> validation Dice izle
-> early stopping uygula
-> best checkpoint'i kaydet
```

Keras tabanli modellerde loss:

```text
0.5 * BCE + 0.5 * Dice Loss
```

PyTorch tabanli modellerde ana loss:

```text
BCEWithLogits + Dice Loss
```

Model ek head urettiyse edge/aux/image kayiplari yalnizca ilgili modelin
destekledigi durumda kullanildi.

## 8. Threshold secimi

Threshold yine test setinden secilmedi.

```text
validation probability map
-> 0.10-0.90 arasi threshold tarama
-> en iyi validation threshold secimi
-> internal-test setine sabit uygulama
```

Image-level score olarak probability map'in maksimum degeri kullanildi:

```text
image_score = max(probability_map)
```

## 9. Hesaplanan metrikler

Ana seed stability metrigi:

```text
forged-only pixel F1
```

Bu metrik forged goruntulerdeki lokalizasyon basarisini olcer. Authentic
goruntulerde GT maskeler sifir oldugu icin all-image pixel ortalamalari tek
basina model lokalizasyonunu anlatmakta zayif kalabilir.

Ek metrikler:

```text
forged-only pixel IoU
forged-only precision
forged-only recall
image F1
image ROC-AUC
epochs ran
selected threshold
```

## 10. Seed bazli sonuc ozeti

Uc seed ortalamasina gore sonuc:

| Model | Forged pixel F1 mean | Std | Forged IoU mean | Image F1 mean | Image ROC-AUC mean |
|---|---:|---:|---:|---:|---:|
| SegFormer-B0 | 0.2666 | 0.0585 | 0.1815 | 0.6777 | 0.7375 |
| EfficientNetB0-UNet | 0.1866 | 0.0077 | 0.1272 | 0.5608 | 0.5991 |
| U-Net++ | 0.0528 | 0.0311 | 0.0329 | 0.3367 | 0.5298 |
| DeepLabV3+ | 0.0283 | 0.0114 | 0.0174 | 0.2402 | 0.5549 |
| Plain U-Net | 0.0255 | 0.0343 | 0.0149 | 0.2195 | 0.5182 |

Yorum:

```text
SegFormer-B0 en guclu ortalama performansi verdi.
EfficientNetB0-UNet daha dusuk ama daha kararli bir CNN baseline oldu.
Plain U-Net, U-Net++ ve DeepLabV3+ bu pilot kararlilik testinde ana aday
olacak kadar guclu gorunmedi.
```

## 11. DeepLabV3+ caveat'i

Deney 2 sirasinda internal lightweight DeepLabV3+ fallback hattinda ASPP
image-pooling branch'i ile BatchNorm kaynakli batch-size edge case goruldu.
Sorun, 1x1 feature map uzerinde BatchNorm kullanilmasindan kaynaklandi.

Bu nedenle Deney 2'deki DeepLabV3+ sonuclari, duzeltilmis current-code
referansi olarak okunmalidir. Eski DeepLabV3+ kosulari ile yeni fallback
sonuclarini karistirmamak gerekir.

## 12. Deney 3/4'e gecis gerekcesi

Deney 2, SegFormer-B0'nin seed degisiminde de guclu kaldigini ve
EfficientNetB0-UNet'in kararli bir CNN baseline oldugunu gosterdi.

Fakat Deney 2 hala kucuk pilot subset uzerindeydi. Sonraki deneylerde daha
buyuk/full split, girdi temsili ve model ailesi karsilastirmalari yapildi.

## 13. Kaynak dosyalar

```text
deney_2/pilot_experiment_design.md
deney_2/pilot_experiment_interpretation.md
deney_2/pilot_seed_comparison.csv
deney_2/pilot_seed_per_run_results.csv
deney_2/pilot_seed_42/{model}/
deney_2/pilot_seed_123/{model}/
deney_2/pilot_seed_2025/{model}/
analysis_review/experiment_1_2_fairness_notes.md
```
