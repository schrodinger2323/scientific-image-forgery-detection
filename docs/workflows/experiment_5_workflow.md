# Deney 5 Adim Adim Islem Akisi

Bu not, `recod_luc_experiment5_calibration_postprocessing.py` kodu ve
`deney_5/experiments_4_full/experiment5_calibration_postprocessing` altindaki
CSV/JSON ciktilari esas alinarak hazirlanmistir. Amac, Deney 5'te tam olarak
hangi islemlerin hangi sirayla yapildigini ve secimlerin neye gore
belirlendigini aciklamaktir.

## 1. Deney 5'in temel amaci

Deney 5 yeni bir model egitimi degildir. Deney 4'te egitilmis dort modelin
validation ve test probability map ciktilari kullanilarak karar katmani
iyilestirilmistir.

Kullanilan modeller:

| Model klasoru | Kisa ad |
|---|---|
| `unetpp_resnet34_rgb_full` | U-Net++ ResNet34 |
| `efficientnetb0_unet_rgb_full` | EfficientNetB0-UNet |
| `segformer_b0_rgb_full` | SegFormer-B0 |
| `dinov2_lite_decoder_rgb_full` | DINOv2-lite decoder |

Girdi olarak her goruntu icin modelin urettigi `probability map` alinir. Bu
harita 0-1 araliginda piksel bazli "sahte bolge olasiligi" gibi dusunulur.
Deney 5 bu harita uzerinden:

1. pixel threshold secer,
2. post-processing uygular,
3. goruntu duzeyinde skor ve threshold secer,
4. validation metriklerine gore strateji secer,
5. secilen config'i test setine aynen uygular.

Test seti hicbir parametre seciminde kullanilmamistir.

## 2. Kullanilan veri bolumleri

Deney 5, ortak split yapisini kullanir:

| Split | Toplam | Authentic | Forged |
|---|---:|---:|---:|
| train | 3590 | 1665 | 1925 |
| validation | 515 | 240 | 275 |
| test | 1023 | 472 | 551 |

Train/validation/test arasinda `image_id` overlap yoktur. Deney 5'te secimler
validation setinde yapilir, nihai raporlama test setindedir.

## 3. Genel hiperparametre arama uzayi

Kodda Deney 5 icin taranan ana adaylar:

| Parametre | Aday degerler |
|---|---|
| `pixel_threshold` | 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75 |
| `image_threshold` | 0.01-0.99 arasi 0.02 adimlarla + image score quantile adaylari |
| `min_component_area` | 0, 25, 100, 500 |
| `min_component_mean_probability` | 0.0, 0.2, 0.4 |
| `morphology` | none, open, close, open_close |
| `morph_kernel_size` | 3, 5 |
| `top_k_components` | None, 1, 2, 3 |
| `component_iou_thresholds` | 0.10, 0.25, 0.50 |
| `image_score_type` | max_probability, top1_mean_probability, top5_mean_probability, pred_mask_ratio_raw, pred_mask_ratio_clean, max_component_mean_probability, max_component_area_ratio |

`image_size=256` oldugu icin `min_component_area=500`, 256x256 evaluation
uzayinda 500 piksel demektir. Bu yaklasik `500 / 65536 = 0.0076`, yani
goruntunun yaklasik %0.76'sidir. Bu deger orijinal goruntu pikseli degildir.

## 4. Tek bir config bir goruntuye nasil uygulanir?

Bir config ornegi:

```text
pixel_threshold = 0.75
postprocess_mode = morph_area_probability_clean
min_component_area = 500
min_component_mean_probability = 0.0
morphology = open
morph_kernel_size = 5
image_score_type = max_probability
```

Bu config her validation goruntusune tek tek uygulanir.

### 4.1 Probability map'ten raw mask uretimi

Ilk adim binary maske uretmektir:

```text
raw_mask = probability_map >= pixel_threshold
```

Bu adimdan sonra `raw_mask`, 0/1 degerli tahmin maskesidir.

### 4.2 Post-processing icin connected component hesabi

Post-processing sirasinda connected component'ler sadece tahmin maskesi
uzerinde hesaplanir. Ground truth bu asamada kullanilmaz.

Islem:

```text
raw_mask veya morphology uygulanmis tahmin maskesi
-> cv2.connectedComponentsWithStats(..., connectivity=8)
-> tahmin component listesi
```

Her predicted component icin:

```text
area
area_ratio = area / image_pixel_count
mean_probability = component icindeki probability map ortalamasi
max_probability = component icindeki en yuksek probability
```

Bu bilgiler daha sonra component temizleme icin kullanilir.

## 5. Post-processing modlari nasil calisir?

Post-processing, `raw_mask` uzerinden `final_mask` ureten asamadir.

### 5.1 raw

Temizlik yoktur.

```text
final_mask = raw_mask
```

### 5.2 min_area_clean

Component alani belirlenen esikten kucukse silinir.

```text
keep component if component.area >= min_component_area
```

Ornegin `min_component_area=500` ise 500 pikselden kucuk predicted component'ler
silinir. Bu deger validation grid search ile secilir; elle sabitlenmis tek
deger degildir.

### 5.3 probability_gated

Component'in ortalama olasiligi dusukse silinir.

```text
keep component if component.mean_probability >= min_component_mean_probability
```

Kodda denenen adaylar `0.0`, `0.2`, `0.4` olmustur. Yani `0.2` secilirse
ortalama probability'si 0.2'den dusuk component'ler silinir; `0.4` secilirse
0.4'ten dusuk olanlar silinir.

Final secilen ana adaylarda saf `probability_gated` modu secilmemistir. Nonzero
mean probability filtresi belirgin olarak U-Net++ `low_false_alarm`
konfigurasyonunda `0.4` ile secilmistir. SegFormer, EfficientNetB0 ve DINOv2
final dengeli adaylarinda `min_component_mean_probability=0.0` oldugu icin
pratikte probability gate devrede degildir.

### 5.4 area_probability_clean

Alan ve ortalama olasilik birlikte kullanilir.

```text
keep if area >= min_component_area
and mean_probability >= min_component_mean_probability
```

### 5.5 morph_area_probability_clean

Once morphology uygulanir, sonra component filtreleri calisir.

```text
raw_mask
-> morphology(open / close / open_close)
-> connected components
-> area + mean probability filtering
-> final_mask
```

Deney 5'te final siralamanin en ustundeki adaylar genellikle bu kalibi
kullanmistir:

```text
pixel_threshold = 0.75
postprocess_mode = morph_area_probability_clean
min_component_area = 500
min_component_mean_probability = 0.0
morphology = open
morph_kernel_size = 5
```

### 5.6 keep_topk_components

Alan ve probability filtrelerinden sonra component'ler `area` veya
`mean_probability` ile siralanir; sadece en iyi K component tutulur.

```text
sort components by area or mean_probability
keep top K
```

## 6. Goruntu duzeyinde image score nasil hesaplanir?

Post-processing sonrasinda her goruntu icin tek bir image-level skor uretilir.
Bu skor, goruntunun forged olup olmadigina karar vermek icin kullanilir.

| Score type | Hesaplama |
|---|---|
| `max_probability` | Probability map'teki en yuksek piksel degeri |
| `top1_mean_probability` | En yuksek olasilikli %1 pikselin ortalamasi |
| `top5_mean_probability` | En yuksek olasilikli %5 pikselin ortalamasi |
| `pred_mask_ratio_raw` | Raw mask'teki pozitif piksel orani |
| `pred_mask_ratio_clean` | Final post-processed mask'teki pozitif piksel orani |
| `max_component_mean_probability` | Final component'ler icindeki en yuksek mean probability |
| `max_component_area_ratio` | Final component'ler icindeki en buyuk alan orani |

Goruntu duzeyi karar:

```text
image_pred = image_score >= image_threshold
```

`image_threshold` validation setinde secilir. Kod image F1'i maksimize eden
threshold'u arar; esitliklerde recall ve specificity dikkate alinir. Bu islem
model agirliklarini degistirmez. Buradaki calibration, modelin yeniden
kalibre edilmesi degil, saved probability map uzerinden karar threshold'unun
validation setinde ayarlanmasidir.

## 7. Calibration metrikleri neyi olcer?

Image-level calibration metrikleri `image_score` ve `image_label` uzerinden
hesaplanir:

```text
y_true = image_label
y_score = image_score
```

Hesaplanan image-level metrikler:

```text
accuracy
precision
recall / sensitivity
specificity
F1
ROC-AUC
AUPRC
Brier score
ECE 10-bin
confusion matrix: TP, FN, TN, FP
```

ECE hesabi:

1. Image score 0-1 araligina clip edilir.
2. 10 confidence bin olusturulur.
3. Her bin icin ortalama confidence ve empirical forged oranina bakilir.
4. Agirlikli mutlak farklar toplanir.

```text
ECE = sum_bin (n_bin / n_total) * abs(empirical_accuracy_bin - avg_confidence_bin)
```

Bu calibration bolumu image-level karar kalitesini olcer. Segmentation maskesinin
kendisini degistiren kisim post-processing bolumudur.

## 8. Bir config validation setinde nasil puanlanir?

Her aday config validation setindeki tum goruntulere uygulanir. Sonra tek bir
validation metrik satiri uretilir.

### 8.1 Pixel-level metrikler

Pixel metrikleri tum ilgili pikseller uzerinden TP/FP/TN/FN toplayarak
hesaplanir.

```text
Dice = 2TP / (2TP + FP + FN)
IoU = TP / (TP + FP + FN)
```

`val_forged_dice`, validation'daki forged goruntulerin pikselleri uzerinde
aggregate olarak hesaplanir. Bu, "her goruntunun Dice'i hesaplanip ortalamasi
alindi" anlamina gelmez. Kod once forged validation goruntulerindeki tum
piksel-level TP/FP/FN degerlerini toplar, sonra tek bir Dice hesaplar.

### 8.2 Per-image metrikler

Her goruntu icin ayri Dice/IoU da kaydedilir:

```text
gt_area
gt_area_ratio
pred_area
pred_area_ratio
dice
iou
image_score
image_pred_label
```

Bu per-image tablo ozellikle small-object analysis ve failure case incelemeleri
icin kullanilir.

### 8.3 Component-aware metrikler

Component-aware evaluation post-processing'den farklidir. Burada hem GT mask
hem de final prediction mask uzerinde connected component cikarilir.

```text
GT mask -> GT components
final prediction mask -> predicted components
```

Sonra GT-pred component IoU matrisi kurulur:

```text
component_iou = intersection(gt_component, pred_component) / union(gt_component, pred_component)
```

Eslestirme icin Hungarian matching kullanilir:

```text
linear_sum_assignment(-iou_matrix)
```

Bir eslesme TP sayilmak icin IoU threshold'unu gecmelidir. Kullanilan threshold
degerleri:

```text
0.10, 0.25, 0.50
```

Sonra:

```text
component_precision = TP / (TP + FP)
component_recall = TP / (TP + FN)
component_F1 = 2PR / (P + R)
```

### 8.4 Authentic false alarm

Authentic false alarm image-level forged tahmini degildir. Authentic bir
goruntude post-processing sonrasinda pozitif predicted component kalip
kalmadigina bakar.

```text
authentic_fp_rate =
authentic goruntulerde pred_component_count > 0 olanlar / authentic goruntu sayisi
```

Bu nedenle `image_specificity` ve `authentic_fp_rate` ayni sey degildir.

## 9. Validation grid search iki asamada nasil yapildi?

### 9.1 Stage 1

Stage 1'de daha basit config'ler denenir:

```text
pixel_threshold
image_score_type
raw
min_area_clean
min_component_area
```

Arama boyutu:

```text
7 pixel threshold
x 7 image score type
x (1 raw + 3 min_area_clean)
= 196 config
```

Bu config'ler `balanced_score`, `val_forged_dice`,
`val_component_f1_iou010` siralamasina gore siralanir.

### 9.2 Stage 2

Stage 1'de en iyi ilk `stage1_top_k=5` config alinir. Bunlarin etrafinda daha
zengin post-processing varyantlari denenir:

```text
probability_gated
area_probability_clean
morph_area_probability_clean
keep_topk_components
```

Tum validation sonuclari model klasorlerindeki su dosyalara yazilir:

```text
{model}/val_grid_search_all.csv
{model}/val_grid_search_top50.csv
```

## 10. Stratejiler nasil secildi?

Grid search sonucunda her model icin validation metrik satirlari olusur.
Strateji secimi bu satirlar arasindan yapilir; tek tek goruntu secilmez.

| Strateji | Secim kurali |
|---|---|
| `best_forged_dice` | `val_forged_dice` en yuksek config; esitlikte `val_forged_iou` |
| `best_component_f1` | `val_component_f1_iou010` en yuksek config; esitlikte `val_authentic_fp_rate` dusuk |
| `balanced_final_score` | `balanced_score` en yuksek config; esitlikte `val_forged_dice` |
| `small_object_focused` | validation'da en kucuk %25 forged maskelerde mean per-image Dice en yuksek config |
| `low_false_alarm` | `val_authentic_fp_rate <= 0.25` olanlar icinde component F1 en yuksek config; yoksa en dusuk authentic FP'li ilk 10 icinde forged Dice en yuksek config |

Validation balanced score:

```text
balanced_score =
0.35 * val_forged_dice
+ 0.25 * val_component_f1_iou010
+ 0.20 * val_image_f1
+ 0.20 * (1 - val_authentic_fp_rate)
```

Ornek: `best_forged_dice` icin sistem tek tek "en iyi goruntuyu" secmez.
Her config tum validation forged goruntulerine uygulanir; forged pikseller
uzerinde tek bir aggregate Dice hesaplanir. `val_forged_dice` en yuksek olan
config, `best_forged_dice` stratejisi olarak kaydedilir.

## 11. Test asamasi nasil yapildi?

Validation'da secilen config'ler test setine aynen uygulanir. Testte yeniden
threshold secimi, yeniden post-processing secimi veya yeniden strateji secimi
yapilmaz.

Her model icin test edilen config'ler:

```text
5 validation stratejisi + 1 raw_reference = 6 config
```

Dort model oldugu icin:

```text
4 model x 6 config = 24 test sonucu
```

Test final score:

```text
final_score =
0.35 * test_dice_forged_only
+ 0.25 * component_f1_iou010
+ 0.20 * image_f1
+ 0.20 * (1 - authentic_fp_rate)
```

## 12. Secilen onemli config'ler

Final siralamada ust siradaki adaylar:

| Rank | Model | Strateji | Post-process | Pixel thr | Image score | Image thr | Min area | Mean prob | Test forged Dice | Comp F1@0.10 | Image F1 | Auth FP | Final |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | SegFormer-B0 | low_false_alarm | morph_area_probability_clean | 0.75 | max_probability | 0.5388 | 500 | 0.0 | 0.5573 | 0.3263 | 0.7292 | 0.2436 | 0.5737 |
| 2 | SegFormer-B0 | balanced_final_score | morph_area_probability_clean | 0.75 | max_probability | 0.5388 | 500 | 0.0 | 0.5573 | 0.3263 | 0.7292 | 0.2436 | 0.5737 |
| 3 | EfficientNetB0-UNet | balanced_final_score | morph_area_probability_clean | 0.75 | max_probability | 0.7049 | 500 | 0.0 | 0.5199 | 0.3251 | 0.7400 | 0.1992 | 0.5714 |
| 4 | DINOv2-lite | balanced_final_score | morph_area_probability_clean | 0.75 | max_probability | 0.3100 | 500 | 0.0 | 0.5293 | 0.3283 | 0.7032 | 0.2394 | 0.5601 |
| 5 | DINOv2-lite | low_false_alarm | min_area_clean | 0.75 | max_probability | 0.3100 | 500 | 0.0 | 0.5296 | 0.3292 | 0.7032 | 0.2415 | 0.5600 |
| 6 | EfficientNetB0-UNet | low_false_alarm | min_area_clean | 0.65 | max_probability | 0.7049 | 500 | 0.0 | 0.5065 | 0.3201 | 0.7400 | 0.2542 | 0.5544 |

Bu tablo sunu gosterir: final dengeli/false alarm odakli adaylarda `500`
piksel alani siklikla secilmistir. Ancak bu deger butun stratejiler icin sabit
degildir; ornegin `best_forged_dice` veya `small_object_focused` stratejilerinde
25, 100 veya raw config'ler de secilmistir.

## 13. Kisa yontem ozeti

Deney 5'te islem sirasi sudur:

```text
Deney 4 probability map
-> pixel threshold
-> raw binary mask
-> post-processing ile final mask
-> image score hesaplama
-> validation'da image threshold secimi
-> pixel / image / component / authentic false alarm metrikleri
-> validation strateji secimi
-> secilen config'in test setine sabit uygulanmasi
```

Bu nedenle Deney 5'in katkisi, model mimarisini degistirmek degil; mevcut
probability map ciktilarini validation tabanli karar kurallariyla daha
kullanilabilir hale getirmektir. En belirgin kazanim, ozellikle balanced ve
low-false-alarm stratejilerinde kucuk/guvensiz predicted component'leri
temizleyerek authentic goruntulerde sahte pozitif component oranini azaltmaktir.

## 14. Kaynak dosyalar

Ana kod:

```text
recod_luc_experiment5_calibration_postprocessing.py
```

Ana cikti klasoru:

```text
deney_5/experiments_4_full/experiment5_calibration_postprocessing
```

Baslica kanit/cikti dosyalari:

```text
experiment5_config.json
split_summary.csv
selected_configs_all_models.csv
test_results_all_strategies.csv
final_candidate_ranking.csv
{model}/val_grid_search_all.csv
{model}/selected_configs.csv
{model}/test_results_by_strategy.csv
{model}/test_per_image_metrics_{strategy}.csv
{model}/test_component_details_{strategy}.csv
{model}/image_level_calibration_{strategy}.csv
{model}/small_mask_bin_metrics_{strategy}.csv
```
