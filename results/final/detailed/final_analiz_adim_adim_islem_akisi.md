# Final Analiz Adim Adim Islem Akisi

Bu not, `recod_luc_final_analysis.py` kodu ve
`final_analysis/final_analysis` altindaki ciktilar okunarak hazirlanmistir.
Amac, final analizde hangi islemlerin hangi sirayla yapildigini, hangi
metriklerin nasil hesaplandigini ve tezde hangi caveat'lerin korunmasi
gerektigini aciklamaktir.

## 1. Final analizin temel amaci

Final analiz yeni model egitmez. Deney 6 sonrasinda belirlenen iki final aday
modeli clean test, robustness, failure case, gorsel analiz ve istatistiksel
karsilastirma protokolleriyle degerlendirir.

Ana kural:

```text
Test setinde threshold veya post-processing parametresi secilmez.
Deney 6 validation setinde secilmis config clean ve robustness kosullarinda sabit tutulur.
```

Final analizde iki ana model vardir:

| Model | Rol | Kullanilan strateji |
|---|---|---|
| `segformer_b0_rgb_384_smallmask` | best_localization_model | `balanced_final_score` |
| `efficientnetb0_unet_rgb_384_smallmask` | low_false_alarm_model | `low_false_alarm` |

Karsilastirma icin Deney 5/256 referans modelleri de tabloya eklenmistir:

```text
U-Net++ ResNet34 256
EfficientNetB0-UNet 256
SegFormer-B0 256
DINOv2-lite 256
```

## 2. Kullanilan split ve cache siniri

Final analiz ortak split dosyalarini kullanir:

| Split | Toplam | Authentic | Forged |
|---|---:|---:|---:|
| train | 3590 | 1665 | 1925 |
| validation | 515 | 240 | 275 |
| test | 1023 | 472 | 551 |

Leakage kontrolu:

```text
train-val overlap = 0
train-test overlap = 0
val-test overlap = 0
```

Final analiz icin ana cache siniri:

```text
final_analysis/final_analysis
```

Bu klasorde probability map, robustness metrikleri ve downstream tablo/gorseller
varsa pahali inference adimlari yeniden calistirilmeden downstream artifact'ler
yeniden uretilebilir.

## 3. Global final analiz ayarlari

Ana config:

| Parametre | Deger |
|---|---:|
| `seed` | 42 |
| `batch_size` | 4 |
| `num_workers` | 0 |
| `pin_memory` | false |
| `use_amp` | true |
| `save_probability_maps` | true |
| `bootstrap_iterations` | 5000 |
| `robustness_enabled` | true |
| `reuse_existing_robustness_metrics` | true |
| `robustness_include_combined` | true |
| `component_iou_thresholds` | 0.10, 0.25, 0.50 |

Robustness kosullari:

```text
clean_png
jpeg_q90
jpeg_q70
jpeg_q50
gaussian_blur_light
gaussian_blur_medium
gaussian_noise_light
gaussian_noise_medium
combined_jpeg70_blur_light
```

## 4. Final aday config'leri nasil alindi?

Final analiz, Deney 6'da validation setinde secilmis strateji config'lerini
kullanir. Test setinde yeniden secim yapmaz.

SegFormer-B0 384:

```text
strategy = balanced_final_score
pixel_threshold = 0.85
image_score_type = max_probability
image_threshold = 0.59
postprocess_mode = morph_area_probability_clean
min_component_area = 100
min_component_mean_probability = 0.0
morphology = open_close
morph_kernel_size = 5
```

EfficientNetB0-UNet 384:

```text
strategy = low_false_alarm
pixel_threshold = 0.85
image_score_type = max_probability
image_threshold = 0.78
postprocess_mode = morph_area_probability_clean
min_component_area = 100
min_component_mean_probability = 0.0
morphology = open_close
morph_kernel_size = 5
```

Bu config'ler clean test ve robustness bozulmalarinda aynen kullanilir.

## 5. Clean test degerlendirmesi

Clean test asamasinda her final model icin:

```text
checkpoint veya cached test probability map yuklenir
-> Deney 6 validation config'i uygulanir
-> final binary mask uretilir
-> pixel, image, component, small-mask ve calibration metrikleri hesaplanir
```

Probability map varsa cache kullanilabilir. Yoksa checkpoint ile test
goruntulerinden yeniden probability map uretilir.

Clean test sonuc dosyalari:

```text
clean_final_candidate_results.csv
{model}/clean_test_metrics.csv
{model}/test_per_image_metrics_{strategy}.csv
{model}/small_mask_bin_metrics_{strategy}.csv
{model}/test_component_details_{strategy}.csv
{model}/image_level_calibration_{strategy}.csv
```

## 6. Post-processing final analizde nasil uygulanir?

Her test goruntusu icin once probability map'ten raw mask uretilir:

```text
raw_mask = probability_map >= pixel_threshold
```

Sonra Deney 6'da secilen post-processing config'i uygulanir:

```text
raw_mask
-> morphology(open_close, kernel=5)
-> connected components
-> min_component_area filtresi
-> min_component_mean_probability filtresi
-> final_mask
```

Final adaylarda `min_component_area=100` ve
`min_component_mean_probability=0.0` oldugu icin pratikte ana temizlik
morphology + 100 piksel alani uzerinden yapilir. Connected component'ler bu
temizlik asamasinda sadece predicted mask uzerinde hesaplanir; GT mask burada
kullanilmaz.

## 7. Pixel-level ve forged-only metrikler

Pixel metrikleri final predicted mask ve GT mask uzerinden hesaplanir.

```text
Dice = 2TP / (2TP + FP + FN)
IoU = TP / (TP + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
Specificity = TN / (TN + FP)
```

`dice_forged_only`, sadece forged test goruntulerindeki pikseller uzerinde
aggregate olarak hesaplanir. Bu per-image Dice ortalamasi degildir.

Clean test ana sonuc:

| Model | Forged Dice | Forged IoU | Q1 Dice | Q2 Dice |
|---|---:|---:|---:|---:|
| SegFormer-B0 384 | 0.6451 | 0.4761 | 0.2767 | 0.3249 |
| EfficientNetB0-UNet 384 | 0.5770 | 0.4055 | 0.2833 | 0.3174 |

Yorum:

```text
SegFormer-B0 384 aggregate localization'da daha iyi.
EfficientNetB0-UNet 384 Q1 Dice'ta cok az daha yuksek, fakat forged Dice daha dusuk.
```

## 8. Small-mask quartile analizi

Final analiz, test forged goruntulerini `mask_quartile` alanina gore Q1-Q4
gruplarina ayirir. Q1 en kucuk ground-truth sahtecilik maskelerini temsil eder.

Her quartile icin:

```text
n
mean_dice
median_dice
mean_iou
median_iou
dice_lt_005_count
dice_lt_010_count
mean_pred_area_ratio
mean_gt_area_ratio
```

Ana clean degerler:

| Model | Q1 Dice | Q2 Dice | Q3 Dice | Q4 Dice |
|---|---:|---:|---:|---:|
| SegFormer-B0 384 | 0.2767 | 0.3249 | 0.4399 | 0.6922 |
| EfficientNetB0-UNet 384 | 0.2833 | 0.3174 | 0.4127 | 0.5685 |

Scale notu: Q1/Q2 gruplari ground-truth maske alanindan gelir. Bu alan
yorumlanirken original-image `gt_area` ile resized evaluation alanini
karistirmamak gerekir.

## 9. Component-aware evaluation

Component metrikleri post-processing tamamlandiktan sonra hesaplanir.

Islem:

```text
GT mask -> connected components
final prediction mask -> connected components
GT-pred IoU matrisi
Hungarian matching
IoU threshold gecen eslesmeler TP
```

Kullanilan IoU threshold'lari:

```text
0.10, 0.25, 0.50
```

Ana clean component F1@0.10:

| Model | Component F1@0.10 | Component F1@0.25 | Component F1@0.50 |
|---|---:|---:|---:|
| SegFormer-B0 384 | 0.4384 | 0.4165 | 0.3415 |
| EfficientNetB0-UNet 384 | 0.4206 | 0.4014 | 0.3399 |

## 10. Authentic false alarm

`authentic_fp_rate`, authentic test goruntulerinde final predicted maskede
pozitif component kalip kalmadigini olcer.

```text
authentic_fp_rate =
authentic goruntulerde pred_component_count > 0 olanlar / authentic goruntu sayisi
```

Clean test:

| Model | Authentic FP rate | Image specificity |
|---|---:|---:|
| SegFormer-B0 384 | 0.3559 | 0.3602 |
| EfficientNetB0-UNet 384 | 0.2394 | 0.5699 |

Yorum:

```text
EfficientNetB0-UNet 384 daha konservatif modeldir.
SegFormer-B0 384 daha iyi lokalizasyon saglar, fakat authentic false alarm orani daha yuksektir.
```

## 11. Image-level calibration

Image-level karar icin final config'teki `image_score_type=max_probability`
kullanilir.

```text
image_score = max(probability_map)
image_pred = image_score >= image_threshold
```

Image threshold:

| Model | Image threshold |
|---|---:|
| SegFormer-B0 384 | 0.59 |
| EfficientNetB0-UNet 384 | 0.78 |

Hesaplanan image-level metrikler:

```text
image_accuracy
image_precision
image_recall / sensitivity
image_specificity
image_f1
image_roc_auc
image_auprc
image_brier
image_ece_10bin
```

Clean test:

| Model | Image F1 | ROC-AUC | AUPRC | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|
| SegFormer-B0 384 | 0.7620 | 0.8503 | 0.8857 | 0.2639 | 0.2659 |
| EfficientNetB0-UNet 384 | 0.7878 | 0.8616 | 0.8890 | 0.2417 | 0.2578 |

## 12. Robustness analizi

Robustness analizi clean PNG test goruntulerine in-memory bozulmalar uygular.
Maskeler degistirilmez; threshold ve post-processing config'leri clean/validation
secimiyle sabit kalir.

Bozulmalar:

```text
jpeg_q90
jpeg_q70
jpeg_q50
gaussian_blur_light
gaussian_blur_medium
gaussian_noise_light
gaussian_noise_medium
combined_jpeg70_blur_light
```

`reuse_existing_robustness_metrics=True` oldugu icin
`robustness_metrics_all.csv` mevcutsa robustness metrikleri cache'ten okunur.

Ana robustluk ozeti:

| Model | En belirgin bozulma | Forged Dice degisim | Q1 Dice degisim | Auth FP degisim |
|---|---|---:|---:|---:|
| SegFormer-B0 384 | gaussian_noise_medium | -0.0276 | -0.1000 | +0.1589 |
| EfficientNetB0-UNet 384 | jpeg_q50 / gaussian_noise_medium | -0.0095 / -0.0328 | -0.1791 / -0.1462 | +0.0127 / +0.1292 |

Yorum:

```text
Gaussian noise medium iki model icin de zorlayici.
Q1 Dice, JPEG kalite dususu ve noise'a forged Dice'tan daha hassas.
```

## 13. Final model karsilastirma tablosu

Final analiz `final_model_comparison.csv` ve
`tables/final_model_comparison.csv` dosyalarini uretir. Bu tablo 384 final
adaylarini ve 256 referans modellerini ayni metrik kolonlariyla toplar.

Baslica satirlar:

| Model | Strateji | Forged Dice | Q1 Dice | Component F1@0.10 | Auth FP | Image F1 |
|---|---|---:|---:|---:|---:|---:|
| SegFormer-B0 384 | balanced_final_score | 0.6451 | 0.2767 | 0.4384 | 0.3559 | 0.7620 |
| EfficientNetB0-UNet 384 | low_false_alarm | 0.5770 | 0.2833 | 0.4206 | 0.2394 | 0.7878 |
| SegFormer-B0 256 | balanced_final_score | 0.5573 | NA | 0.3263 | 0.2436 | 0.7292 |
| EfficientNetB0-UNet 256 | balanced_final_score | 0.5199 | NA | 0.3251 | 0.1992 | 0.7400 |
| DINOv2-lite 256 | balanced_final_score | 0.5293 | NA | 0.3283 | 0.2394 | 0.7032 |
| U-Net++ ResNet34 256 | raw_reference | 0.5092 | NA | 0.2246 | 0.6780 | 0.6985 |

## 14. Istatistiksel testler

Final analiz per-image metriklerle paired karsilastirmalar yapar.

Kullanilan testler:

```text
paired t-test
Wilcoxon signed-rank test
Cohen's d
rank-biserial effect size
bootstrap mean-difference CI
McNemar image correctness testi
```

Ornek paired sonuclar:

| Karsilastirma | Metrik | Subset | n | Mean diff | Wilcoxon p | Bootstrap CI |
|---|---|---|---:|---:|---:|---|
| SegFormer 384 vs EfficientNet 384 | Dice | forged | 551 | +0.0380 | 8.28e-05 | [0.0220, 0.0548] |
| SegFormer 384 vs EfficientNet 384 | Dice | Q1 | 138 | -0.0066 | 0.1610 | [-0.0354, 0.0209] |
| SegFormer 384 vs SegFormer 256 | Dice | forged | 551 | +0.1533 | 4.10e-33 | [0.1342, 0.1734] |
| SegFormer 384 vs SegFormer 256 | Dice | Q1 | 138 | +0.2444 | 5.66e-09 | [0.1948, 0.2935] |
| EfficientNet 384 vs EfficientNet 256 | Dice | forged | 551 | +0.1271 | 3.04e-20 | [0.1066, 0.1474] |
| EfficientNet 384 vs EfficientNet 256 | Dice | Q1 | 138 | +0.2147 | 7.80e-07 | [0.1652, 0.2637] |

McNemar caveat:

```text
mcnemar_tests.csv dosyasinda n=1967 gorunuyor.
Bu test setindeki 1023 goruntuden buyuk oldugu icin image_id collision veya merge-key sorunu olasi.
Tezde McNemar sonucunu dogrudan kullanmadan once sample_id/case_key ile yeniden dogrulamak gerekir.
```

## 15. Failure case analizi

Her final model icin failure-case CSV ve PNG gridleri uretildi:

```text
best_forged_cases
worst_forged_cases
small_mask_failures
false_positive_authentic
false_negative_forged
large_mask_success
large_mask_failure
```

Her grup icin en fazla 12 ornek kaydedildi. Dosyalar:

```text
{model}/failure_cases/{group}.csv
{model}/failure_cases/{group}.png
```

Ek olarak iki final modelin ayrildigi ornekler de kaydedildi:

```text
model_disagreement_cases.csv
failure_cases/model_disagreement_cases.csv
model_disagreement_cases.png
plots/model_disagreement_examples.png
```

## 16. Uretilen grafikler

Ana grafikler:

```text
plots/final_model_comparison_barplots.png
plots/confusion_matrices_final.png
plots/reliability_diagram_final.png
plots/per_image_dice_distribution.png
plots/resolution_gain_256_vs_384.png
plots/small_mask_quartile_final.png
plots/scatter_q1_dice_vs_authfp_final.png
plots/scatter_forged_dice_vs_image_f1_final.png
plots/roc_pr_curves_final.png
plots/robustness_q1_dice.png
plots/robustness_forged_dice.png
plots/robustness_component_f1.png
plots/robustness_auth_fp_rate.png
```

Bu grafikler clean performans, 256->384 kazanci, small-mask quartile davranisi,
robustness ve calibration yorumlari icin hazirlandi.

## 17. Final karar ozeti

`final_decision_summary.json` dosyasindaki karar:

```text
Ana hedef piksel/bilesen lokalizasyon basarisiysa final model SegFormer-B0 384'tur.
Pratik kullanimda dusuk yanlis alarm ve goruntu duzeyi guvenilirlik oncelikliyse final model EfficientNetB0-UNet 384'tur.
Tezde iki model birlikte raporlanmalidir.
```

Bu ayrim metriklerle uyumludur:

```text
SegFormer-B0 384: daha yuksek forged Dice, IoU ve component F1.
EfficientNetB0-UNet 384: daha dusuk authentic FP, daha yuksek image F1/specificity.
```

## 18. Final analiz akisi kisa ozet

```text
Ortak splitleri oku ve leakage kontrolu yap
-> final aday model config'lerini Deney 6 selected_configs dosyalarindan al
-> clean test probability map/checkpoint yukle
-> validation-secimli threshold ve post-processing'i sabit uygula
-> clean pixel, small-mask, component, image-level calibration metriklerini hesapla
-> robustness bozulmalarini uygula veya cached robustness_metrics_all.csv oku
-> 256 referans modellerini Deney 5/4 CSV'lerinden ekle
-> final_model_comparison tablosunu uret
-> paired istatistiksel testleri ve bootstrap CI'lari hesapla
-> failure case ve model disagreement dosyalarini/gorsellerini olustur
-> final_analysis_report.md ve final_decision_summary.json yaz
```

## 19. Kaynak dosyalar

Ana kod:

```text
recod_luc_final_analysis.py
```

Ana cikti klasoru:

```text
final_analysis/final_analysis
```

Baslica cikti dosyalari:

```text
final_analysis_config.json
split_summary.csv
clean_final_candidate_results.csv
final_model_comparison.csv
tables/final_model_comparison.csv
robustness_metrics_all.csv
robustness_delta_from_clean.csv
statistics/statistical_tests.csv
statistics/bootstrap_confidence_intervals.csv
statistics/mcnemar_tests.csv
final_failure_case_summary.csv
failure_cases/all_failure_cases.csv
failure_cases/model_disagreement_cases.csv
final_analysis_report.md
final_decision_summary.json
```
