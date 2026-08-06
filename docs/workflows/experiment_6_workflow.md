# Deney 6 Adim Adim Islem Akisi

Bu not, `recod_luc_experiment6_smallmask_384.py` kodu ve
`deney_6/experiments_full/experiment6_smallmask_384` altindaki ciktilar
okunarak hazirlanmistir. Amac, Deney 6'da tam olarak hangi islemlerin hangi
sirayla yapildigini, hangi secimlerin validation setinde belirlendigini ve
test setine neyin sabit uygulandigini aciklamaktir.

## 1. Deney 6'nin temel amaci

Deney 6, Deney 5 sonunda hala zayif kalan kucuk sahtecilik bolgelerini
iyilestirmeyi hedefler. Deney 5 daha cok mevcut probability map'ler uzerinde
calibration/post-processing denemesi yaparken, Deney 6'da iki guclu aday model
384x384 cozumurlukte yeniden egitilmistir.

Deney 6 yeni bir model ailesi aramaz. Deney 5 sonunda anlamli kalan iki modeli
odaga alir:

| Model klasoru | Kisa ad | Durum |
|---|---|---|
| `efficientnetb0_unet_rgb_384_smallmask` | EfficientNetB0-UNet 384 | calistirildi |
| `segformer_b0_rgb_384_smallmask` | SegFormer-B0 384 | calistirildi |
| `efficientnetb0_unet_rgb_384_smallmask_oversampling` | EfficientNetB0 384 + oversampling | kodda var, ana kosuda kapali |

Raporlanan ana kosuda oversampling ve small-mask loss weighting kapali
tutulmustur. Bunun nedeni, once 384x384 cozumurluk artisi etkisini izole
etmektir.

## 2. Deney 5'ten farki

Deney 5:

```text
Deney 4 probability map'leri
-> validation'da threshold/post-processing aramasi
-> testte sabit config
```

Deney 6:

```text
Secilen modelleri 384x384 yeniden egit
-> validation probability map uret
-> kucuk maske odakli threshold/post-processing aramasi yap
-> secilen config'i testte sabit uygula
```

Yani Deney 6'da hem egitim cozumurlugu hem de validation strateji skoru
kucuk maske performansini daha fazla dikkate alacak sekilde degistirilmistir.

## 3. Ortak veri ve split yapisi

Deney 6, Deney 4/5 ile ayni ortak split dosyalarini kullanir:

```text
deney_4/_shared_splits_seed42
```

Test seti threshold, image-level threshold, post-processing veya strateji
seciminde kullanilmamistir. Tum secimler validation setinde yapilmis, test
seti sadece final degerlendirme icin kullanilmistir.

## 4. Kucuk maske gruplari nasil tanimlandi?

Her split icin forged goruntulerde ground-truth maske alani hesaplandi.
Authentic goruntulerde `gt_area=0` kabul edildi. Forged goruntuler `gt_area`
degerine gore dort quartile'a ayrildi:

```text
Q1 = en kucuk maskeler
Q2 = ikinci kucuk grup
Q3 = orta-buyuk grup
Q4 = en buyuk maskeler
```

Bu quartile atamasi original-image mask alanina gore yapildi. Daha sonra egitim
ve evaluation sirasinda goruntu/maskeler 384x384'e resize edildi. Bu nedenle
quartile esikleri ile 384x384 evaluation piksel alanlari ayni olcek degildir.

Kaydedilen original-image quartile esikleri:

| Split | Q1 max area | Q2 max area | Q3 max area | Q1/Q2/Q3/Q4 sayilari |
|---|---:|---:|---:|---|
| train | 2637 | 7753 | 29648 | 482 / 481 / 481 / 481 |
| val | 2451 | 6821 | 32990.5 | 69 / 69 / 68 / 69 |
| test | 2493 | 8387 | 30102.5 | 138 / 138 / 137 / 138 |

Bu bilgiler su dosyaya yazildi:

```text
mask_area_quartiles.json
mask_area_summary_train.csv
mask_area_summary_val.csv
mask_area_summary_test.csv
```

## 5. Egitim verisi ve augmentasyon

Her goruntu 384x384'e resize edildi. Maske resize islemi nearest-neighbor ile
yapildi; boylece maske sinirlari interpolasyonla ara degerlere donusmedi.

Train augmentasyonlari:

```text
Resize 384x384
HorizontalFlip
VerticalFlip
RandomRotate90
ShiftScaleRotate
RandomBrightnessContrast
GaussianBlur
GaussNoise
ImageCompression
Normalize(ImageNet mean/std)
```

Validation/test augmentasyonu sadece resize ve normalize icerir.

## 6. Egitim ayarlari

Ortak global ayarlar:

| Parametre | Deger |
|---|---:|
| `seed` | 42 |
| `image_size` | 384 |
| `effective_batch_size` | 8 |
| `epochs` | 40 |
| `learning_rate` | 1e-4 |
| `weight_decay` | 1e-4 |
| `use_amp` | true |
| `early_stopping_patience` | 8 |
| `scheduler_patience` | 3 |
| `scheduler_factor` | 0.5 |
| `run_robustness` | false |

EfficientNetB0-UNet modelinde batch size 8, grad accumulation 1 kullanildi.
SegFormer-B0 modelinde bellek nedeniyle batch size 4, grad accumulation 2
kullanildi. Boylece effective batch size 8 korunmus oldu.

## 7. Loss fonksiyonu

Egitimde BCE + Dice loss kullanildi:

```text
loss = 0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss
```

Pozitif piksel oraninin dusuk olmasi nedeniyle `pos_weight` kullanildi. Kod
train splitindeki toplam piksel ve pozitif maske piksel sayisindan ham agirligi
hesaplar, sonra 1 ile 20 arasina clip eder:

```text
raw_weight = negative_pixels / positive_pixels
pos_weight = clip(raw_weight, 1, 20)
```

Bu kosuda `pos_weight=20.0` olarak kaydedilmistir.

Kodda Q1/Q2 icin loss weight altyapisi vardir:

```text
Q1 -> loss_weight 2.0
Q2 -> loss_weight 1.5
```

Ancak raporlanan ana Deney 6 kosusunda `use_small_mask_loss_weight=False`
oldugu icin bu agirliklar egitim loss'una uygulanmamistir.

## 8. Oversampling neden kapaliydi?

Kodda Q1/Q2 oversampling altyapisi vardir:

```text
Q1 -> sampling weight 3.0
Q2 -> sampling weight 2.0
```

Fakat `efficientnetb0_unet_rgb_384_smallmask_oversampling` konfigurasyonu
`enabled=False` olarak birakilmistir. Ana kosuda oversampling kapali
tutulmustur. Gerekce: Deney 6'nin temel sorusu once "256 yerine 384
cozumurluk kucuk maskeleri iyilestiriyor mu?" oldugu icin ek sampling
mudahalesi devre disi birakildi.

## 9. Model secimi ve checkpoint monitor skoru

Her epoch sonunda validation setinde 0.5 pixel threshold ile per-image Dice
hesaplandi. Forged goruntuler icin:

```text
val_forged_dice_t050
val_q1_dice_t050
val_q2_dice_t050
```

monitor skoru:

```text
val_smallmask_score_t050 =
0.50 * val_forged_dice_t050
+ 0.30 * val_q1_dice_t050
+ 0.20 * val_q2_dice_t050
```

`best_model.pth`, bu monitor skorunu en yuksek yapan epoch'ta kaydedildi.
Scheduler de ayni monitor skoruna gore calisti.

Kayitli egitim ozeti:

| Model | Son epoch | Batch | Grad accum | Pos weight | Training time |
|---|---:|---:|---:|---:|---:|
| SegFormer-B0 384 | 40 | 4 | 2 | 20.0 | 4730.27 sn |
| EfficientNetB0-UNet 384 | 32 | 8 | 1 | 20.0 | 3864.28 sn |

EfficientNetB0 tarafinda early stopping 32. epoch'ta durmustur. SegFormer
40 epoch'a kadar devam etmistir.

## 10. Prediction map uretimi

Egitimden sonra en iyi checkpoint yuklendi. Validation ve test setleri icin
probability map uretildi:

```text
model(image) -> logits -> sigmoid(logits) -> probability map
```

Kaydedilen dosyalar:

```text
{model}/val_prob_maps.npz
{model}/val_metadata.csv
{model}/test_prob_maps.npz
{model}/test_metadata.csv
```

Bundan sonraki threshold/post-processing asamasi bu probability map'ler
uzerinde yapildi.

## 11. Post-processing arama uzayi

Deney 6'da validation grid search icin su adaylar tarandi:

| Parametre | Aday degerler |
|---|---|
| `pixel_threshold` | 0.45, 0.55, 0.65, 0.75, 0.85 |
| `image_threshold` | 0.01-0.99 arasi 0.01 adim |
| `min_component_area` | 0, 25, 100, 500 |
| `min_component_mean_probability` | 0.0, 0.2, 0.4 |
| `morphology` | none, open, close, open_close |
| `morph_kernel_size` | 3, 5 |
| `top_k_components` | None, 1, 2, 3 |
| `image_score_type` | max_probability, top1_mean_probability, top5_mean_probability, pred_mask_ratio_raw, pred_mask_ratio_clean, max_component_mean_probability, max_component_area_ratio |

Post-processing baslangici:

```text
raw_mask = probability_map >= pixel_threshold
```

Sonra post-processing mode'a gore final mask uretilir.

## 12. Post-processing modlari

`raw`:

```text
final_mask = raw_mask
```

`min_area_clean`:

```text
component.area >= min_component_area olanlar tutulur
```

`probability_gated`:

```text
component.mean_probability >= min_component_mean_probability olanlar tutulur
```

`area_probability_clean`:

```text
area filtresi + mean probability filtresi birlikte uygulanir
```

`morph_area_probability_clean`:

```text
raw_mask
-> morphology
-> connected components
-> area + mean probability filtresi
-> final_mask
```

`keep_topk_components`:

```text
component'ler area veya mean_probability ile siralanir
top K component tutulur
```

Post-processing icin connected component'ler sadece predicted mask uzerinde
hesaplanir. Ground truth bu temizlik asamasinda kullanilmaz.

## 13. Image-level score ve calibration

Her final mask/probability map icin tek bir image score hesaplandi:

| Score type | Hesaplama |
|---|---|
| `max_probability` | probability map maksimum piksel degeri |
| `top1_mean_probability` | en yuksek %1 pikselin ortalamasi |
| `top5_mean_probability` | en yuksek %5 pikselin ortalamasi |
| `pred_mask_ratio_raw` | raw mask pozitif piksel orani |
| `pred_mask_ratio_clean` | final mask pozitif piksel orani |
| `max_component_mean_probability` | final component'ler icindeki en yuksek mean probability |
| `max_component_area_ratio` | final component'ler icindeki en buyuk alan orani |

Image-level karar:

```text
image_pred = image_score >= image_threshold
```

`image_threshold`, validation setinde image F1'i maksimize edecek sekilde
secildi. Test setinde yeniden image threshold secilmedi.

Calibration metrikleri image score uzerinden hesaplandi:

```text
image_brier
image_ece_10bin
image_roc_auc
image_auprc
image_f1
image_specificity
```

## 14. Validation grid search nasil yapildi?

Her post-processing config once validation setindeki tum goruntulere uygulandi.
Segmentasyon maskesi image score type'tan bagimsiz oldugu icin once tek
segmentation cikisi hesaplandi; sonra 7 farkli `image_score_type` icin ayri
image score ve image threshold denendi.

Her config icin validation metrikleri:

```text
val_forged_dice
val_small_q1_dice
val_small_q2_dice
val_component_f1_iou010
val_image_f1
val_authentic_fp_rate
```

Validation balanced score:

```text
balanced_score =
0.30 * val_forged_dice
+ 0.25 * val_small_q1_dice
+ 0.20 * val_component_f1_iou010
+ 0.15 * val_image_f1
+ 0.10 * (1 - val_authentic_fp_rate)
```

Small-object score:

```text
small_object_score =
0.40 * val_small_q1_dice
+ 0.25 * val_small_q2_dice
+ 0.20 * val_component_f1_iou010
+ 0.15 * (1 - val_authentic_fp_rate)
```

Tum validation sonuclari:

```text
{model}/val_grid_search_all.csv
{model}/val_grid_search_top50.csv
```

## 15. Stratejiler nasil secildi?

Her model icin validation grid search sonucundan bes strateji secildi:

| Strateji | Secim kurali |
|---|---|
| `best_forged_dice` | `val_forged_dice` maksimum; esitlikte Q1 Dice |
| `best_small_mask_q1_dice` | `val_small_q1_dice` maksimum; esitlikte Q2 Dice |
| `balanced_final_score` | `balanced_score` maksimum; esitlikte Q1 Dice |
| `low_false_alarm` | `val_authentic_fp_rate <= 0.25` olanlar icinde component F1 ve forged Dice iyi olan config |
| `small_object_practical` | `small_object_score` maksimum; esitlikte forged Dice |

Bu stratejiler config satiri secer; tek tek goruntu secmez. Secilen config'ler:

```text
{model}/selected_configs.csv
```

## 16. Test setinde ne yapildi?

Validation'da secilen strateji config'leri test setine aynen uygulandi.
Testte:

```text
threshold yeniden secilmedi
post-processing yeniden secilmedi
image_threshold yeniden secilmedi
model yeniden egitilmedi
```

Her model icin 5 strateji test edildi. Iki model oldugu icin toplam 10 test
satiri olustu.

Kayit edilen test dosyalari:

```text
{model}/test_results_by_strategy.csv
{model}/test_per_image_metrics_{strategy}.csv
{model}/small_mask_bin_metrics_{strategy}.csv
{model}/test_component_details_{strategy}.csv
{model}/image_level_calibration_{strategy}.csv
```

Test final score:

```text
final_score =
0.25 * dice_forged_only
+ 0.25 * q1_dice
+ 0.15 * q2_dice
+ 0.15 * component_f1_iou010
+ 0.10 * image_f1
+ 0.10 * (1 - authentic_fp_rate)
```

Ana ranking dosyasi:

```text
experiment6_final_candidate_ranking.csv
```

## 17. Final secilen onemli config'ler

Final siralamanin ilk satirlari:

| Rank | Model | Strateji | Pixel thr | Image score | Image thr | Post-process | Min area | Mean prob | Morph | Kernel | Forged Dice | Q1 Dice | Q2 Dice | Comp F1@0.10 | Image F1 | Auth FP | Final |
|---:|---|---|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | SegFormer-B0 384 | balanced_final_score | 0.85 | max_probability | 0.59 | morph_area_probability_clean | 100 | 0.0 | open_close | 5 | 0.6451 | 0.2767 | 0.3249 | 0.4384 | 0.7620 | 0.3559 | 0.4856 |
| 2 | SegFormer-B0 384 | small_object_practical | 0.85 | max_probability | 0.59 | morph_area_probability_clean | 0 | 0.0 | open_close | 5 | 0.6451 | 0.2849 | 0.3530 | 0.4179 | 0.7620 | 0.4004 | 0.4843 |
| 3 | SegFormer-B0 384 | best_forged_dice | 0.85 | max_probability | 0.59 | morph_area_probability_clean | 25 | 0.0 | open | 5 | 0.6453 | 0.2850 | 0.3531 | 0.4124 | 0.7620 | 0.4004 | 0.4835 |
| 4 | EfficientNetB0-UNet 384 | balanced_final_score | 0.85 | max_probability | 0.78 | morph_area_probability_clean | 100 | 0.0 | open_close | 5 | 0.5770 | 0.2833 | 0.3174 | 0.4206 | 0.7878 | 0.2394 | 0.4806 |
| 5 | EfficientNetB0-UNet 384 | low_false_alarm | 0.85 | max_probability | 0.78 | morph_area_probability_clean | 100 | 0.0 | open_close | 5 | 0.5770 | 0.2833 | 0.3174 | 0.4206 | 0.7878 | 0.2394 | 0.4806 |

En yuksek final score SegFormer-B0 384 `balanced_final_score` ile geldi.
EfficientNetB0-UNet 384 daha dusuk forged Dice'a sahipti; buna karsilik
authentic false alarm orani daha dusuktu.

## 18. Component-aware evaluation

Component-aware evaluation, post-processing'den sonra final prediction mask ve
ground truth mask uzerinde yapildi.

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

Raporlanan ana component metriği genellikle `component_f1_iou010` olmustur.

## 19. Failure case ve gorsel analiz

Balanced strategy icin her modelde prediction grid ve failure case dosyalari
uretildi:

```text
prediction_examples_best_strategy.png
q1_best_cases.png
q1_worst_cases.png
q1_false_negative_cases.png
failure_cases/small_mask_failures.csv
failure_cases/large_mask_failures.csv
failure_cases/false_positive_authentic.csv
failure_cases/false_negative_forged.csv
failure_cases/best_cases_forged.csv
```

Bu dosyalar kucuk maske hatalarini, buyuk maske hatalarini, authentic false
positive'leri ve false negative forged orneklerini incelemek icin olusturuldu.

## 20. Kisa yontem ozeti

Deney 6'da akış su sekildedir:

```text
Ortak splitleri oku
-> forged mask alanlarindan Q1-Q4 kucuk maske gruplarini tanimla
-> EfficientNetB0-UNet ve SegFormer-B0 modellerini 384x384 egit
-> best checkpoint'i small-mask monitor skoruyla sec
-> validation/test probability map uret
-> validation'da post-processing + image score + image threshold grid search yap
-> bes strateji config'i sec
-> secilen config'leri test setine sabit uygula
-> pixel, small-mask, component, image-level calibration ve false alarm metriklerini raporla
-> final score ile adaylari sirala
```

Deney 6'nin metodolojik yorumu: Bu deney, Deney 5'teki karar katmani
iyilestirmesinin otesine gecip 384x384 yeniden egitimle kucuk maskelerin daha
fazla piksel temsili kazanip kazanmadigini test eder. Ek oversampling/loss
weighting ana kosuda kapali tutuldugu icin raporlanan etki esas olarak
cozumurluk artisi ve small-mask odakli validation seciminden gelir.

## 21. Kaynak dosyalar

Ana kod:

```text
recod_luc_experiment6_smallmask_384.py
```

Ana cikti klasoru:

```text
deney_6/experiments_full/experiment6_smallmask_384
```

Baslica cikti dosyalari:

```text
experiment6_config.json
mask_area_quartiles.json
mask_area_summary_train.csv
mask_area_summary_val.csv
mask_area_summary_test.csv
experiment6_all_results.csv
test_results_all_strategies.csv
experiment6_final_candidate_ranking.csv
experiment6_vs_experiment5_comparison.csv
experiment6_vs_experiment5_comparison.md
{model}/metrics.csv
{model}/config.json
{model}/best_model.pth
{model}/val_prob_maps.npz
{model}/test_prob_maps.npz
{model}/val_grid_search_all.csv
{model}/selected_configs.csv
{model}/test_results_by_strategy.csv
{model}/small_mask_bin_metrics_{strategy}.csv
{model}/test_component_details_{strategy}.csv
{model}/image_level_calibration_{strategy}.csv
```
