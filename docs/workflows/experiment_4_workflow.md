# Deney 4 Adim Adim Islem Akisi

Bu not, `analysis_review/experiment_4_full_analysis.md`,
`analysis_review/deney4_kod_ve_sonuc_ozeti.md`,
`analysis_review/experiment_4_model_comparison_key_metrics.csv` ve
`deney_4/experiments_4_full/*` altindaki model artefaktlari okunarak
hazirlanmistir.

## 1. Deneyin temel amaci

Deney 4, tam ReCodAI-LUC split'i uzerinde dort RGB segmentasyon modelini ayni
egitim ve validation-secimli evaluation protokoluyle karsilastirdi.

Calistirilan modeller:

| Model | Deney klasoru | Mimari |
|---|---|---|
| U-Net++ ResNet34 | `unetpp_resnet34_rgb_full` | CNN encoder-decoder baseline |
| EfficientNetB0-UNet | `efficientnetb0_unet_rgb_full` | parametre-verimli CNN baseline |
| SegFormer-B0 | `segformer_b0_rgb_full` | transformer semantic segmentation |
| DINOv2-lite decoder | `dinov2_lite_decoder_rgb_full` | foundation feature + hafif decoder |

Deney 3'te RGB-only yaklasim ana Dice/IoU tarafinda daha guclu kaldigi icin
Deney 4'te butun modeller RGB input ile calistirildi.

## 2. Veri kumesi ve shared split

Deney 4 tam veri split'ini kullandi:

| Split | Toplam | Authentic | Forged |
|---|---:|---:|---:|
| Train | 3590 | 1665 | 1925 |
| Validation | 515 | 240 | 275 |
| Test | 1023 | 472 | 551 |
| Full | 5128 | 2377 | 2751 |

Split `image_id` group anahtariyla seed 42 kullanilarak kuruldu. Leakage
kontrolunde train-val, train-test ve val-test overlap degerleri 0 raporlandi.

Onemli caveat:

```text
Ayni image_id authentic ve forged tarafinda birlikte bulunabildigi icin
istatistiksel merge islemlerinde tek basina image_id guvenli degildir.
```

Bu nedenle duzeltilmis analizlerde `image_path` veya daha guvenli sample key
kullanildi.

## 3. Maske okuma ve veri temsili

Veri akisi:

```text
train_images/authentic ve train_images/forged indeksle
-> forged goruntu icin train_masks/{image_id}.npy ara
-> ayni image_id ile birden fazla maske varsa binary union yap
-> authentic icin sifir mask uret
-> image ve maskeyi input/evaluation boyutuna resize et
-> image'i ImageNet mean/std ile normalize et
```

Maskelerde:

```text
(H, W) -> mask > 0
(C, H, W) -> np.any(mask > 0, axis=0)
(H, W, C) -> np.any(mask > 0, axis=-1)
```

Mask resize icin nearest-neighbor kullanildi.

## 4. Model ciktilari

Tum modeller sigmoid uygulanmamis logit uretir.

Evaluation sirasinda:

```text
logits
-> sigmoid(logits)
-> probability_map
-> validation'da secilen pixel threshold
-> binary mask
```

DINOv2 modeli patch uyumlulugu nedeniyle 252 input kullandi; cikislar ortak
256 evaluation boyutuna yeniden olceklendi.

## 5. Egitim protokolu

Ortak ayarlar:

| Ayar | Deger |
|---|---|
| Seed | 42 |
| Evaluation image size | 256 |
| DINOv2 input size | 252 |
| Batch size | genel 8, DINOv2 4 |
| Epoch ust siniri | 40 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau |
| Early stopping patience | 8 |
| Loss | `0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss` |
| AMP | CUDA varsa acik |

Train augmentasyonlari Albumentations ile yapildi. Validation ve testte sadece
resize + normalization uygulandi.

Best checkpoint validation forged Dice'a gore `best_model.pth` olarak kaydedildi.

## 6. Evaluate-only devam kosusu

Ilk kosuda modeller egitildi ve checkpoint'ler olustu. Ancak PyTorch 2.6+
tarafinda `torch.load` varsayilaninin `weights_only=True` olmasi checkpoint
yuklemede sorun cikardi.

Bu nedenle:

```text
recod_luc_4model_evaluate_existing_checkpoints.py
```

dosyasi kullanildi. Bu script modeli yeniden egitmedi. Var olan
`best_model.pth` dosyalarini `weights_only=False` ile yukledi ve eksik kalan
validation threshold search + test evaluation adimlarini tamamladi.

## 7. Threshold secimi

Test set threshold secimi icin kullanilmadi.

Pixel threshold aramasi:

```text
0.10 - 0.90 arasi, 0.05 adim
```

Image threshold aramasi:

```text
0.00 - 1.00 arasi, 0.01 adim
```

Bu kosuda tum modellerde image score yontemi `max_probability` olarak secildi.

Secilen ayarlar:

| Model | Best epoch | Pixel threshold | Image threshold | Min component area |
|---|---:|---:|---:|---:|
| SegFormer-B0 | 35 | 0.85 | 0.54 | 100 |
| DINOv2-lite decoder | 23 | 0.70 | 0.34 | 200 |
| EfficientNetB0-UNet | 28 | 0.75 | 0.70 | 50 |
| U-Net++ ResNet34 | 10 | 0.70 | 0.79 | 25 |

## 8. Raw ve simple clean post-processing

Raw mod:

```text
probability_map >= selected_pixel_threshold
```

Simple clean mod:

```text
raw binary mask
-> connected component'leri cikar
-> min_component_area altindaki component'leri sil
-> opsiyonel mean probability filtresi uygula
-> final clean mask
```

Deney 4'te clean mod forged Dice'i buyuk olcude degistirmedi; etkisi daha cok
component sayisi, authentic false positive ve specificity tarafinda okundu.

## 9. Pixel-level metrikler

Pixel metrikleri iki seviyede hesaplandi:

```text
all: authentic + forged tum test goruntuleri
forged_only: sadece forged test goruntuleri
```

Ana lokalizasyon yorumu forged-only Dice, forged-only IoU ve forged AUPRC
uzerinden yapildi.

Ana test sonuclari:

| Model | Forged Dice | Forged IoU | Forged AUPRC | Image F1 |
|---|---:|---:|---:|---:|
| SegFormer-B0 | 0.5686 | 0.3972 | 0.6020 | 0.7283 |
| DINOv2-lite decoder | 0.5279 | 0.3586 | 0.5502 | 0.7019 |
| EfficientNetB0-UNet | 0.5216 | 0.3528 | 0.5591 | 0.7399 |
| U-Net++ ResNet34 | 0.5092 | 0.3416 | 0.5268 | 0.6983 |

## 10. Image-level calibration

Image-level skor:

```text
image_score = max(probability_map)
```

Validation setinde image threshold secildi ve testte sabit uygulandi.

Confusion davranisi:

| Model | TP | FN | TN | FP | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2-lite decoder | 544 | 7 | 17 | 455 | 0.9873 | 0.0360 |
| SegFormer-B0 | 512 | 39 | 129 | 343 | 0.9292 | 0.2733 |
| EfficientNetB0-UNet | 502 | 49 | 168 | 304 | 0.9111 | 0.3559 |
| U-Net++ ResNet34 | 442 | 109 | 199 | 273 | 0.8022 | 0.4216 |

DINOv2-lite cok yuksek recall verirken specificity cok dusuk kaldigi icin
image-level kalibrasyon problemi tasir.

## 11. Component-aware evaluation

Component-aware evaluation egitimin parcasi degildir; test evaluation katmanidir.

Islem:

```text
final prediction mask ve GT mask uzerinden connected components cikar
-> GT/pred component IoU matrisi kur
-> Hungarian matching ile eslestir
-> IoU threshold'u gecen eslesmeleri TP say
-> eslesmeyen pred component FP, eslesmeyen GT component FN
```

Kullanilan IoU threshold'lari:

```text
0.10 ve 0.25
```

IoU 0.10 ozet:

| Model | Component precision | Component recall | Component F1 | Authentic FP rate |
|---|---:|---:|---:|---:|
| SegFormer-B0 | 0.4120 | 0.3161 | 0.3577 | 0.3051 |
| DINOv2-lite decoder | 0.3800 | 0.3129 | 0.3432 | 0.4174 |
| EfficientNetB0-UNet | 0.3008 | 0.3870 | 0.3385 | 0.4682 |
| U-Net++ ResNet34 | 0.2328 | 0.4049 | 0.2956 | 0.7182 |

## 12. Kucuk maske analizi

Forged test goruntuleri GT maske alanina gore ceyrekliklere ayrildi.

En kucuk alan grubu 256x256 evaluation uzayinda yaklasik:

```text
<= 533.5 px
```

Bu grup tum modeller icin zordu. Q1 mean Dice:

| Model | Q1 mean Dice | Q1 median Dice | Dice < 0.05 |
|---|---:|---:|---:|
| EfficientNetB0-UNet | 0.1914 | 0.0000 | 79 |
| U-Net++ ResNet34 | 0.1784 | 0.0000 | 80 |
| SegFormer-B0 | 0.0812 | 0.0000 | 117 |
| DINOv2-lite decoder | 0.0685 | 0.0000 | 110 |

Bu bulgu Deney 5 ve Deney 6'nin motivasyonunu olusturdu: full-data modeller
genel Dice'ta rekabetci olsa da kucuk sahtecilik bolgeleri hala zayifti.

## 13. Istatistiksel test caveat'i

Orijinal `deney_4/experiments_full/statistical_tests.csv` dosyasi sadece
`image_id` ile merge yaptigi icin test satir sayisini 1023 yerine 1967'ye
sisirdi.

Duzeltilmis analizlerde `image_path` anahtariyla 1023 test goruntusu uzerinden
yeniden hesaplama yapildi. Bu caveat tezde korunmalidir.

## 14. Deney 5'e gecis gerekcesi

Deney 4 sonunda en guclu genel model SegFormer-B0 oldu. EfficientNetB0-UNet
per-image denge ve image F1 acisindan guclu bir baseline verdi.

Fakat Deney 4'te:

```text
authentic false positive oranlari yuksek kaldi
kucuk maskelerde Dice cok dusuktu
simple clean post-processing sinirli etki gosterdi
```

Bu nedenle Deney 5'te yeni egitim yapmak yerine Deney 4 probability map'leri
uzerinden daha sistematik threshold, image-level calibration, component temizleme
ve strateji secimi yapildi.

## 15. Kaynak dosyalar

```text
recod_luc_4model_full_study.ipynb
recod_luc_4model_evaluate_existing_checkpoints.py
deney_4/_shared_splits_seed42/{full,train,val,test}.csv
deney_4/experiments_4_full/*/summary.json
deney_4/experiments_4_full/*/test_metrics.csv
deney_4/experiments_4_full/*/test_metrics_raw.csv
deney_4/experiments_4_full/*/test_metrics_clean.csv
deney_4/experiments_4_full/*/test_per_image_metrics.csv
deney_4/experiments_4_full/*/test_component_metrics.csv
deney_4/experiments_full/model_comparison_full.csv
analysis_review/experiment_4_full_analysis.md
analysis_review/deney4_kod_ve_sonuc_ozeti.md
analysis_review/experiment_4_* CSV dosyalari
```
