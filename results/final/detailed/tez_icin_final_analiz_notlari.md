# Tez Icin Final Analiz Olcum ve Sonuc Notlari

Bu not, `recod_luc_final_analysis.py` ve `recod_luc_final_analysis.ipynb` ile uretilen `final_analysis/experiments_full/final_analysis` ciktilarini tez yaziminda kullanilabilecek sekilde ozetler. Sayisal degerler okunabilirlik icin genellikle 4 ondaliga yuvarlanmistir; tam hassasiyetli degerler ilgili CSV dosyalarinda korunmaktadir.

## 1. Deneyin amaci ve veri protokolu

Final analiz yeni model egitmez. Deney 6 sonrasinda secilen iki final aday modeli, ayni test bolumu uzerinde clean PNG, JPEG/blur/noise dagilim kaymasi, kucuk maske basarimi, bilesen yakalama, goruntu duzeyi karar ve failure case acisindan degerlendirilir.

Temel kural: threshold ve post-processing parametreleri test setinde secilmemistir. Parametreler onceki validation secimlerinden okunmus, test ve robustness kosullarinda sabit tutulmustur.

Split ozeti:

| split | toplam | authentic | forged |
|---|---:|---:|---:|
| train | 3590 | 1665 | 1925 |
| val | 515 | 240 | 275 |
| test | 1023 | 472 | 551 |

Leakage kontrolu:

| kontrol | ortak image_id |
|---|---:|
| train-val | 0 |
| train-test | 0 |
| val-test | 0 |

Kaynak dosya: `split_summary.csv`.

## 2. Final aday modeller ve secilen stratejiler

| model | rol | image size | strateji | parametre sayisi |
|---|---|---:|---|---:|
| SegFormer-B0 384 | localization-oriented final model | 384 | `balanced_final_score` | 3,714,401 |
| EfficientNetB0-UNet 384 | conservative low-false-alarm final model | 384 | `low_false_alarm` | 6,251,469 |

Clean test degerlendirmesi iki model icin de cached probability map uzerinden yapilmistir (`clean_status=loaded_cached_probability_maps`). Robustness ileri gecisleri icin checkpoint bulunmustur.

Kaynak dosyalar: `clean_final_candidate_results.csv`, `model_runtime_info.csv`.

## 3. Post-processing ve karar esikleri

Iki final modelde ayni piksel ve post-processing yapisi kullanilmistir:

| model | pixel threshold | image threshold | postprocess | min component area | morphology | kernel | image score |
|---|---:|---:|---|---:|---|---:|---|
| SegFormer-B0 384 | 0.85 | 0.59 | `morph_area_probability_clean` | 100 | `open_close` | 5 | `max_probability` |
| EfficientNetB0-UNet 384 | 0.85 | 0.78 | `morph_area_probability_clean` | 100 | `open_close` | 5 | `max_probability` |

Kodda olcum akisi:

1. Probability map piksel bazinda `prob >= pixel_threshold` ile ikili maskeye cevrilir.
2. `open_close` morfoloji islemi 5x5 kernel ile uygulanir.
3. Bagli bilesenler 8-komsulukla cikarilir.
4. Alani 100 pikselden kucuk bilesenler elenir.
5. Goruntu duzeyi skor `max_probability` olarak hesaplanir.
6. Goruntu sahte/gercek karari `image_score >= image_threshold` ile verilir.

## 4. Kodda neyi nasil olctuk?

Piksel duzeyi Dice ve IoU:

`Dice = (2 * TP) / (2 * TP + FP + FN)` ve `IoU = TP / (TP + FP + FN)` olarak hesaplandi. Kodda `EPS=1e-7` ile sifira bolme engellendi. `dice_forged_only` ve `iou_forged_only`, sadece forged goruntulerin tum pikselleri birlestirilerek hesaplanan aggregate lokalizasyon skorudur. `mean_dice_forged` ise her forged goruntu icin Dice hesaplanip ortalamasi alinan per-image skordur; bu ikisi ayni sey degildir.

Kucuk maske analizi:

Her split icinde sadece forged maskelerin `gt_area` degerleri kullanilarak Q1-Q4 maske alan ceyrekleri olusturuldu. Her ceyrek icin mean/median Dice, mean/median IoU ve `dice < 0.05` olan basarisiz vaka sayisi raporlandi.

Bilesen bazli metrikler:

Ground truth ve tahmin maskeleri bagli bilesenlere ayrildi. Bilesenler arasi IoU matrisi cikarildi ve Hungarian matching ile eslestirme yapildi. IoU esikleri 0.10, 0.25 ve 0.50 icin component precision, recall ve F1 hesaplandi. Ana yorumda `component_f1_iou010` daha yumusak nesne yakalama olcutu olarak kullanildi.

Authentic false alarm:

Authentic goruntulerde post-processing sonrasi en az bir tahmin bileseni varsa o goruntu false alarm kabul edildi. `authentic_fp_rate = alarm veren authentic goruntu sayisi / authentic goruntu sayisi`.

Goruntu duzeyi metrikler:

Goruntu skoru `max_probability`, karar esigi ise modelin validation secimli `image_threshold` degeridir. Accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, AUPRC, Brier score ve 10-bin Expected Calibration Error hesaplandi.

Robustness:

Clean PNG testine ek olarak goruntuler bellekte bozuldu, maskeler degistirilmedi ve threshold ayarlari yeniden secilmedi. Kosullar: `jpeg_q90`, `jpeg_q70`, `jpeg_q50`, `gaussian_blur_light` (3x3, sigma 0.8), `gaussian_blur_medium` (5x5, sigma 1.2), `gaussian_noise_light` (std 0.02), `gaussian_noise_medium` (std 0.05), `combined_jpeg70_blur_light`.

Istatistiksel testler:

Per-image Dice/IoU farklari eslenmis olarak karsilastirildi. Paired t-test, Wilcoxon testi, Cohen's d, rank-biserial etki buyuklugu ve 5000 iterasyonlu bootstrap ortalama fark guven araligi hesaplandi.

## 5. Clean test ana sonuclari

| model | forged Dice | forged IoU | mean forged Dice | median forged Dice | Q1 Dice | Q2 Dice | Q3 Dice | Q4 Dice | Comp F1@0.10 | authentic FP | image F1 | ROC-AUC | AUPRC | specificity | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 384 | 0.6451 | 0.4761 | 0.4334 | 0.5257 | 0.2767 | 0.3249 | 0.4399 | 0.6922 | 0.4384 | 0.3559 | 0.7620 | 0.8503 | 0.8857 | 0.3602 | 0.9528 |
| EfficientNetB0-UNet 384 | 0.5770 | 0.4055 | 0.3954 | 0.4797 | 0.2833 | 0.3174 | 0.4127 | 0.5685 | 0.4206 | 0.2394 | 0.7878 | 0.8616 | 0.8890 | 0.5699 | 0.8893 |

Yorum:

SegFormer-B0 384 aggregate forged Dice ve IoU'da daha iyi lokalizasyon verdi. EfficientNetB0-UNet 384 ise authentic false alarm oranini belirgin bicimde daha dusuk tuttu ve goruntu duzeyi F1/ROC-AUC/AUPRC degerlerinde ondeydi. Q1 kucuk maske Dice degeri iki modelde cok yakin; EfficientNet Q1'de 0.0066 puan yuksek, ancak bu fark istatistiksel olarak anlamli degildir.

## 6. 256 referans modellere gore final tablo

| model | image size | strateji | forged Dice | forged IoU | Comp F1@0.10 | authentic FP | image F1 | ROC-AUC | AUPRC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 384 | 384 | `balanced_final_score` | 0.6451 | 0.4761 | 0.4384 | 0.3559 | 0.7620 | 0.8503 | 0.8857 |
| EfficientNetB0-UNet 384 | 384 | `low_false_alarm` | 0.5770 | 0.4055 | 0.4206 | 0.2394 | 0.7878 | 0.8616 | 0.8890 |
| SegFormer-B0 256 | 256 | `balanced_final_score` | 0.5573 | 0.3863 | 0.3263 | 0.2436 | 0.7292 | 0.7613 | 0.8128 |
| DINOv2-lite 256 | 256 | `balanced_final_score` | 0.5293 | 0.3599 | 0.3283 | 0.2394 | 0.7032 | 0.7364 | 0.8031 |
| EfficientNetB0-UNet 256 | 256 | `balanced_final_score` | 0.5199 | 0.3512 | 0.3251 | 0.1992 | 0.7400 | 0.7995 | 0.8446 |
| U-Net++ ResNet34 256 | 256 | `raw_reference` | 0.5092 | 0.3415 | 0.2246 | 0.6780 | 0.6985 | 0.6991 | 0.7377 |

Yorum:

384x384 kucuk maske odakli egitim final modelleri, 256x256 referanslara gore lokalizasyon metriklerinde belirgin artis sagladi. SegFormer-B0 384, tum modeller icinde en yuksek aggregate forged Dice/IoU ve component F1@0.10 degerine ulasti.

## 7. Kucuk maske sonuclari

Test forged maskelerinin alan ceyrekleri:

| quartile | n | min area | max area | mean area | mean area ratio |
|---|---:|---:|---:|---:|---:|
| Q1 | 138 | 158 | 2,488 | 1,054.6 | 0.0111 |
| Q2 | 138 | 2,498 | 8,387 | 4,906.8 | 0.0336 |
| Q3 | 137 | 8,484 | 30,018 | 15,777.7 | 0.0749 |
| Q4 | 138 | 30,187 | 925,845 | 158,824.3 | 0.1007 |

Model bazli kucuk maske basarimi:

| model | quartile | n | mean Dice | median Dice | mean IoU | median IoU | Dice<0.05 |
|---|---|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 384 | Q1 | 138 | 0.2767 | 0.2053 | 0.1957 | 0.1144 | 65 |
| SegFormer-B0 384 | Q2 | 138 | 0.3249 | 0.3358 | 0.2386 | 0.2018 | 53 |
| SegFormer-B0 384 | Q3 | 137 | 0.4399 | 0.5266 | 0.3291 | 0.3574 | 35 |
| SegFormer-B0 384 | Q4 | 138 | 0.6922 | 0.7363 | 0.5770 | 0.5827 | 10 |
| EfficientNetB0-UNet 384 | Q1 | 138 | 0.2833 | 0.2320 | 0.2003 | 0.1313 | 64 |
| EfficientNetB0-UNet 384 | Q2 | 138 | 0.3174 | 0.3204 | 0.2303 | 0.1909 | 57 |
| EfficientNetB0-UNet 384 | Q3 | 137 | 0.4127 | 0.4821 | 0.2990 | 0.3176 | 33 |
| EfficientNetB0-UNet 384 | Q4 | 138 | 0.5685 | 0.5737 | 0.4320 | 0.4022 | 11 |

Yorum:

Kucuk maskeler hala ana zorluk kaynagidir. Q1'de ortalama Dice iki model icin 0.28 civarindadir ve Q1 vakalarinin yaklasik yarisi `Dice < 0.05` seviyesinde kalmistir. Buna karsilik maske alani buyudukce basari belirgin artmakta, ozellikle SegFormer-B0 384 Q4'te 0.6922 mean Dice ve 0.7363 median Dice vermektedir.

## 8. Robustness sonuclari

SegFormer-B0 384:

| kosul | forged Dice | Q1 Dice | Comp F1@0.10 | authentic FP | image F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| clean PNG | 0.6451 | 0.2767 | 0.4384 | 0.3559 | 0.7620 | 0.8503 |
| JPEG90 | 0.6438 | 0.2733 | 0.4349 | 0.3602 | 0.7543 | 0.8443 |
| JPEG70 | 0.6424 | 0.2496 | 0.4226 | 0.4301 | 0.7401 | 0.8208 |
| JPEG50 | 0.6346 | 0.2203 | 0.4078 | 0.4470 | 0.7391 | 0.8043 |
| blur light | 0.6505 | 0.2718 | 0.4398 | 0.3496 | 0.7651 | 0.8476 |
| blur medium | 0.6502 | 0.2606 | 0.4361 | 0.3178 | 0.7638 | 0.8420 |
| noise light | 0.6367 | 0.2567 | 0.4177 | 0.4280 | 0.7390 | 0.8179 |
| noise medium | 0.6175 | 0.1767 | 0.3629 | 0.5148 | 0.7294 | 0.7627 |
| JPEG70 + blur light | 0.6453 | 0.2459 | 0.4242 | 0.4068 | 0.7411 | 0.8227 |

EfficientNetB0-UNet 384:

| kosul | forged Dice | Q1 Dice | Comp F1@0.10 | authentic FP | image F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| clean PNG | 0.5770 | 0.2833 | 0.4206 | 0.2394 | 0.7878 | 0.8616 |
| JPEG90 | 0.5715 | 0.2690 | 0.4104 | 0.2585 | 0.7793 | 0.8517 |
| JPEG70 | 0.5726 | 0.1732 | 0.3923 | 0.2521 | 0.7624 | 0.8302 |
| JPEG50 | 0.5675 | 0.1042 | 0.3640 | 0.2521 | 0.7512 | 0.8057 |
| blur light | 0.5796 | 0.2649 | 0.4186 | 0.2246 | 0.7854 | 0.8595 |
| blur medium | 0.5734 | 0.2348 | 0.4115 | 0.2076 | 0.7752 | 0.8512 |
| noise light | 0.5569 | 0.2186 | 0.3852 | 0.3220 | 0.7643 | 0.8132 |
| noise medium | 0.5442 | 0.1371 | 0.3397 | 0.3686 | 0.7335 | 0.7548 |
| JPEG70 + blur light | 0.5749 | 0.1665 | 0.3930 | 0.2267 | 0.7732 | 0.8360 |

Yorum:

JPEG ve noise bozulmalari ozellikle Q1 kucuk maskeleri dusurmektedir. SegFormer-B0 384, aggregate forged Dice acisindan bozulmalara daha dayanikli gorunur; fakat authentic false alarm orani noise medium kosulunda 0.5148'e kadar cikar. EfficientNetB0-UNet 384, clean ve bircok bozulma kosulunda daha dusuk authentic FP korur; buna karsin Q1 Dice JPEG70/JPEG50 ve combined kosullarinda daha sert duser.

## 9. Istatistiksel karsilastirma ozeti

Final iki model karsilastirmasi:

| karsilastirma | metrik | subset | n | mean diff | 95% bootstrap CI | paired t p | Wilcoxon p |
|---|---|---|---:|---:|---|---:|---:|
| SegFormer 384 - EfficientNet 384 | Dice | forged | 551 | +0.0380 | [0.0220, 0.0548] | 6.30e-06 | 8.28e-05 |
| SegFormer 384 - EfficientNet 384 | IoU | forged | 551 | +0.0448 | [0.0302, 0.0596] | 7.35e-09 | 6.63e-06 |
| SegFormer 384 - EfficientNet 384 | Dice | Q1 | 138 | -0.0066 | [-0.0354, 0.0209] | 0.6434 | 0.1610 |
| SegFormer 384 - EfficientNet 384 | Dice | Q2 | 138 | +0.0076 | [-0.0218, 0.0364] | 0.6129 | 0.5292 |

384 cozumurluk etkisi:

| karsilastirma | metrik | subset | mean diff | 95% bootstrap CI | paired t p |
|---|---|---|---:|---|---:|
| SegFormer 384 - SegFormer 256 | Dice | forged | +0.1533 | [0.1342, 0.1734] | 5.97e-43 |
| SegFormer 384 - SegFormer 256 | Dice | Q1 | +0.2444 | [0.1948, 0.2935] | 3.59e-17 |
| EfficientNet 384 - EfficientNet 256 | Dice | forged | +0.1271 | [0.1066, 0.1474] | 2.17e-29 |
| EfficientNet 384 - EfficientNet 256 | Dice | Q1 | +0.2147 | [0.1652, 0.2637] | 6.37e-14 |

Yorum:

SegFormer-B0 384 ile EfficientNetB0-UNet 384 arasinda genel forged lokalizasyon icin SegFormer lehine anlamli fark vardir. Ancak en kucuk maske ceyreginde (Q1) iki model arasindaki fark anlamli degildir. 384x384 final egitimleri, kendi 256x256 referanslarina gore forged ve Q1 Dice metriklerinde istatistiksel olarak cok guclu artis gostermistir.

Not: `mcnemar_tests.csv` dosyasinda goruntu duzeyi eslesme `n=1967` gorunmektedir; test seti 1023 goruntudur. Kodda McNemar icin merge anahtari `image_id` oldugundan authentic/forged image_id cakismalarinda satir cogalmasi olusabilir. Tezde McNemar sonucunu kullanmadan once `sample_id` veya `case_key` ile yeniden dogrulamak daha guvenlidir. Per-image forged Dice/IoU testlerinde `n=551` beklenen test forged sayisiyla uyumludur.

## 10. Failure case ve gorsel analiz

Her final model icin 7 failure/success grubu uretilmistir ve her grupta en fazla 12 ornek vardir:

| grup | anlam |
|---|---|
| `best_forged_cases` | forged goruntulerde en yuksek Dice ornekleri |
| `worst_forged_cases` | forged goruntulerde en dusuk Dice ornekleri |
| `small_mask_failures` | Q1/Q2 ve Dice<0.05 kucuk maske hatalari |
| `false_positive_authentic` | authentic goruntulerde tahmin bileseni uretilen ornekler |
| `false_negative_forged` | forged oldugu halde goruntu duzeyinde kacirilan ornekler |
| `large_mask_success` | Q4 buyuk maske basarili ornekleri |
| `large_mask_failure` | Q4 buyuk maske basarisiz ornekleri |

Ek olarak `model_disagreement_cases.csv/png`, SegFormer'in EfficientNet'e gore cok daha iyi oldugu 12 ve EfficientNet'in SegFormer'e gore cok daha iyi oldugu 12 vakayi saklar. En buyuk SegFormer lehine farkta Dice farki +0.8976, en buyuk EfficientNet lehine farkta Dice farki -0.7144'tur.

## 11. Tez icin ana karar cumlesi

Ana hedef piksel/bilesen lokalizasyon basarisiysa final model SegFormer-B0 384 olarak raporlanmalidir. Pratik kullanimda dusuk yanlis alarm ve goruntu duzeyi guvenilirlik oncelikliyse EfficientNetB0-UNet 384 daha muhafazakar final adayidir. Tezde iki model birlikte verilmelidir; cunku biri lokalizasyon kalitesini, digeri false alarm maliyetini optimize eden iki farkli operasyonel onceligi temsil etmektedir.

## 12. Kaynak cikti dosyalari

- `final_analysis_report.md`: otomatik uretilen genel final rapor.
- `clean_final_candidate_results.csv`: iki final modelin clean test metrikleri ve secilen config degerleri.
- `final_model_comparison.csv`: 384 final modeller + 256 referanslar ana karsilastirma tablosu.
- `robustness_metrics_all.csv`: clean/JPEG/blur/noise metrikleri.
- `robustness_delta_from_clean.csv`: bozulma kosullarinin clean'e gore farklari.
- `small_mask_bin_metrics_*.csv`: model bazli Q1-Q4 kucuk maske metrikleri.
- `statistical_tests.csv` ve `bootstrap_confidence_intervals.csv`: paired test ve bootstrap guven araliklari.
- `failure_cases/` ve model altindaki `failure_cases/`: tezde kullanilabilecek gorsel ornek ve CSV listeleri.
