# Final Analiz Raporu

## 1. Final analizin amaci

Bu analiz Deney 6 sonrasi iki final aday modeli yeni egitim yapmadan clean test, robustness, failure case ve istatistiksel karsilastirma protokolleriyle degerlendirir.

## 2. Onceki deneylerin kisa ozeti

Deney 5 calibration ve post-processing ile authentic false alarm oranini dusurmus, fakat kucuk sahtecilik bolgelerinde Q1 performansi sinirli kalmistir. Deney 6, EfficientNetB0-UNet ve SegFormer-B0 modellerini 384x384 cozumurlukte egiterek Q1/Q2 performansini belirgin artirmistir.

## 3. Final aday modeller

- `SegFormer-B0 384 balanced`: Localization-oriented final model.

- `EfficientNetB0-UNet 384 balanced / low false alarm`: Conservative low-false-alarm final model.

## 4. PNG goruntulerde JPEG robustness testinin gerekcesi

Bilimsel gorseller makale, PDF, sunum ve web sureclerinde yeniden kaydedilebilir. Bu nedenle PNG test goruntuleri kalici olarak degistirilmeden in-memory JPEG encode/decode ile bozulur; maskeler degistirilmez ve threshold ayarlari clean validation secimiyle sabit tutulur.

## 5. Ana final karsilastirma tablosu

| model_name              | source_model_name                     | role                    |   image_size | strategy             |   forged_dice |   forged_iou |    q1_dice |    q2_dice |    q3_dice |    q4_dice |   dice_lt_005_count |   component_f1_iou010 |   component_f1_iou025 |   component_f1_iou050 |   authentic_fp_rate |   image_f1 |   image_roc_auc |   image_auprc |   brier_score |   ece_10_bins |   inference_time_per_image |   trainable_params | final_interpretation                                                                                      |
|:------------------------|:--------------------------------------|:------------------------|-------------:|:---------------------|--------------:|-------------:|-----------:|-----------:|-----------:|-----------:|--------------------:|----------------------:|----------------------:|----------------------:|--------------------:|-----------:|----------------:|--------------:|--------------:|--------------:|---------------------------:|-------------------:|:----------------------------------------------------------------------------------------------------------|
| SegFormer-B0 384        | segformer_b0_rgb_384_smallmask        | best_localization_model |          384 | balanced_final_score |      0.645101 |     0.476124 |   0.276685 |   0.324935 |   0.439855 |   0.692177 |                 163 |              0.438405 |              0.416544 |              0.341507 |            0.355932 |   0.761974 |        0.850272 |      0.885686 |      0.263876 |      0.265866 |                  0.0450994 |                nan | Localization-oriented final model; pixel/component lokalizasyonu onceliklidir.                            |
| EfficientNetB0-UNet 384 | efficientnetb0_unet_rgb_384_smallmask | low_false_alarm_model   |          384 | low_false_alarm      |      0.576974 |     0.405456 |   0.283306 |   0.31737  |   0.412657 |   0.568471 |                 165 |              0.420648 |              0.401411 |              0.339853 |            0.239407 |   0.787781 |        0.861565 |      0.889    |      0.241748 |      0.257791 |                  0.0497742 |                nan | Conservative low-false-alarm final model; pratik kullanimda yanlis alarm maliyeti dusunulerek raporlanir. |
| SegFormer-B0 256        | segformer_b0_rgb_full                 | 256_reference           |          256 | balanced_final_score |      0.557297 |     0.386287 | nan        | nan        | nan        | nan        |                 288 |              0.326267 |              0.303989 |              0.230686 |            0.243644 |   0.729211 |        0.761262 |      0.812832 |      0.29063  |      0.267255 |                nan         |                nan | 256x256 referans/baseline; yeniden egitilmedi, Deney 5/4 CSV sonucundan okundu.                           |
| DINOv2-lite 256         | dinov2_lite_decoder_rgb_full          | 256_reference           |          256 | balanced_final_score |      0.529252 |     0.359853 | nan        | nan        | nan        | nan        |                 287 |              0.328258 |              0.301522 |              0.193836 |            0.239407 |   0.703155 |        0.736423 |      0.803058 |      0.318782 |      0.305158 |                nan         |                nan | 256x256 referans/baseline; yeniden egitilmedi, Deney 5/4 CSV sonucundan okundu.                           |
| EfficientNetB0-UNet 256 | efficientnetb0_unet_rgb_full          | 256_reference           |          256 | balanced_final_score |      0.519882 |     0.351244 | nan        | nan        | nan        | nan        |                 278 |              0.325147 |              0.310619 |              0.242823 |            0.199153 |   0.74003  |        0.799481 |      0.844583 |      0.289768 |      0.295516 |                nan         |                nan | 256x256 referans/baseline; yeniden egitilmedi, Deney 5/4 CSV sonucundan okundu.                           |
| U-Net++ ResNet34 256    | unetpp_resnet34_rgb_full              | 256_baseline            |          256 | raw_reference        |      0.509167 |     0.341532 | nan        | nan        | nan        | nan        |                 189 |              0.224613 |              0.206357 |              0.155501 |            0.677966 |   0.698512 |        0.699068 |      0.737694 |      0.303105 |      0.274235 |                nan         |                nan | 256x256 referans/baseline; yeniden egitilmedi, Deney 5/4 CSV sonucundan okundu.                           |

## 6. 256x256 vs 384x384 cozumurluk etkisi

| comparison                                                              | metric   | subset   |   n |   mean_a |    mean_b |   mean_diff |   paired_t_stat |   paired_t_p |   wilcoxon_stat |   wilcoxon_p |   cohens_d |   rank_biserial |
|:------------------------------------------------------------------------|:---------|:---------|----:|---------:|----------:|------------:|----------------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| segformer_b0_rgb_384_smallmask vs efficientnetb0_unet_rgb_384_smallmask | dice     | forged   | 551 | 0.433401 | 0.39542   |  0.0379816  |        4.56048  |  6.29641e-06 |           42565 |  8.2824e-05  |  0.194283  |       0.210882  |
| segformer_b0_rgb_384_smallmask vs efficientnetb0_unet_rgb_384_smallmask | iou      | forged   | 551 | 0.335119 | 0.290359  |  0.0447606  |        5.87441  |  7.35298e-09 |           40920 |  6.62806e-06 |  0.250258  |       0.241379  |
| segformer_b0_rgb_384_smallmask vs efficientnetb0_unet_rgb_384_smallmask | dice     | Q1       | 138 | 0.276685 | 0.283306  | -0.00662034 |       -0.463977 |  0.6434      |            2344 |  0.160975    | -0.0394964 |      -0.157592  |
| segformer_b0_rgb_384_smallmask vs efficientnetb0_unet_rgb_384_smallmask | dice     | Q2       | 138 | 0.324935 | 0.31737   |  0.00756511 |        0.507079 |  0.612915    |            2438 |  0.529187    |  0.0431654 |       0.0717685 |
| segformer_b0_rgb_384_smallmask vs segformer_b0_rgb_full                 | dice     | forged   | 551 | 0.433401 | 0.280098  |  0.153303   |       15.0138   |  5.96689e-43 |           31217 |  4.09699e-33 |  0.63961   |       0.589455  |
| segformer_b0_rgb_384_smallmask vs segformer_b0_rgb_full                 | iou      | forged   | 551 | 0.335119 | 0.211926  |  0.123193   |       15.5311   |  2.27224e-45 |           30981 |  1.90857e-33 |  0.661647  |       0.592559  |
| segformer_b0_rgb_384_smallmask vs segformer_b0_rgb_full                 | dice     | Q1       | 138 | 0.276685 | 0.0323276 |  0.244358   |        9.66633  |  3.58713e-17 |            2054 |  5.66051e-09 |  0.822852  |       0.571682  |
| segformer_b0_rgb_384_smallmask vs segformer_b0_rgb_full                 | dice     | Q2       | 138 | 0.324935 | 0.205271  |  0.119664   |        6.67738  |  5.57926e-10 |            2658 |  5.55083e-06 |  0.568416  |       0.44573   |
| efficientnetb0_unet_rgb_384_smallmask vs efficientnetb0_unet_rgb_full   | dice     | forged   | 551 | 0.39542  | 0.268308  |  0.127112   |       11.942    |  2.17171e-29 |           41576 |  3.04116e-20 |  0.508747  |       0.453221  |
| efficientnetb0_unet_rgb_384_smallmask vs efficientnetb0_unet_rgb_full   | iou      | forged   | 551 | 0.290359 | 0.194608  |  0.0957511  |       11.6843   |  2.50382e-28 |           41868 |  6.2802e-20  |  0.49777   |       0.449381  |
| efficientnetb0_unet_rgb_384_smallmask vs efficientnetb0_unet_rgb_full   | dice     | Q1       | 138 | 0.283306 | 0.0686349 |  0.214671   |        8.35827  |  6.3681e-14  |            2471 |  7.80251e-07 |  0.711503  |       0.484725  |
| efficientnetb0_unet_rgb_384_smallmask vs efficientnetb0_unet_rgb_full   | dice     | Q2       | 138 | 0.31737  | 0.212043  |  0.105327   |        5.69283  |  7.29318e-08 |            3249 |  0.00101345  |  0.484606  |       0.32249   |
| segformer_b0_rgb_384_smallmask vs unetpp_resnet34_rgb_full              | dice     | forged   | 551 | 0.433401 | 0.326815  |  0.106586   |       11.4211   |  2.94656e-27 |           38213 |  4.64465e-24 |  0.486554  |       0.497449  |
| segformer_b0_rgb_384_smallmask vs unetpp_resnet34_rgb_full              | iou      | forged   | 551 | 0.335119 | 0.234249  |  0.10087    |       12.3124   |  6.13061e-31 |           37178 |  2.64532e-25 |  0.524524  |       0.51106   |
| segformer_b0_rgb_384_smallmask vs unetpp_resnet34_rgb_full              | dice     | Q1       | 138 | 0.276685 | 0.163525  |  0.11316    |        5.81172  |  4.13928e-08 |            2782 |  1.87501e-05 |  0.494726  |       0.419873  |
| segformer_b0_rgb_384_smallmask vs unetpp_resnet34_rgb_full              | dice     | Q2       | 138 | 0.324935 | 0.267996  |  0.0569397  |        3.16091  |  0.00193635  |            3835 |  0.0412167   |  0.269075  |       0.200292  |
| efficientnetb0_unet_rgb_384_smallmask vs unetpp_resnet34_rgb_full       | dice     | forged   | 551 | 0.39542  | 0.326815  |  0.0686048  |        7.22319  |  1.70159e-12 |           51070 |  2.41974e-11 |  0.307718  |       0.328362  |
| efficientnetb0_unet_rgb_384_smallmask vs unetpp_resnet34_rgb_full       | iou      | forged   | 551 | 0.290359 | 0.234249  |  0.0561092  |        7.32314  |  8.6618e-13  |           50333 |  6.187e-12   |  0.311976  |       0.338055  |
| efficientnetb0_unet_rgb_384_smallmask vs unetpp_resnet34_rgb_full       | dice     | Q1       | 138 | 0.283306 | 0.163525  |  0.11978    |        6.02362  |  1.48461e-08 |            2506 |  1.13961e-06 |  0.512765  |       0.477427  |
| efficientnetb0_unet_rgb_384_smallmask vs unetpp_resnet34_rgb_full       | dice     | Q2       | 138 | 0.31737  | 0.267996  |  0.0493746  |        2.80784  |  0.00571619  |            4174 |  0.186545    |  0.239019  |       0.129601  |

## 7. SegFormer-B0 384 analizi

SegFormer-B0 384 forged Dice=0.6451, Q1 Dice=0.2767, Component F1@0.10=0.4384, authentic FP=0.3559.

## 8. EfficientNetB0-UNet 384 analizi

EfficientNetB0-UNet 384 forged Dice=0.5770, Q1 Dice=0.2833, Component F1@0.10=0.4206, authentic FP=0.2394.

## 9. Localization vs false alarm trade-off

SegFormer-B0 384 genel lokalizasyon odakli model olarak; EfficientNetB0-UNet 384 ise daha muhafazakar yanlis alarm profili icin birlikte raporlanmalidir.

## 10. Robustness sonuclari

| model_name                            | degradation                |   forged_dice |   forged_iou |   q1_dice |   q2_dice |   q3_dice |   q4_dice |   dice_lt_005_count |   component_f1_iou010 |   component_f1_iou025 |   component_f1_iou050 |   authentic_fp_rate |   image_f1 |   image_roc_auc |   image_auprc |   image_specificity |   image_recall |   inference_time_per_image |
|:--------------------------------------|:---------------------------|--------------:|-------------:|----------:|----------:|----------:|----------:|--------------------:|----------------------:|----------------------:|----------------------:|--------------------:|-----------:|----------------:|--------------:|--------------------:|---------------:|---------------------------:|
| segformer_b0_rgb_384_smallmask        | clean_png                  |      0.645101 |     0.476124 |  0.276685 |  0.324935 |  0.439855 |  0.692177 |                 163 |              0.438405 |              0.416544 |              0.341507 |            0.355932 |   0.761974 |        0.850272 |      0.885686 |            0.360169 |       0.952813 |                  0.0450994 |
| segformer_b0_rgb_384_smallmask        | jpeg_q90                   |      0.643791 |     0.474699 |  0.273299 |  0.317839 |  0.440683 |  0.692098 |                 163 |              0.43486  |              0.418316 |              0.336189 |            0.360169 |   0.75431  |        0.84432  |      0.882986 |            0.330508 |       0.952813 |                  0.0554331 |
| segformer_b0_rgb_384_smallmask        | jpeg_q70                   |      0.642403 |     0.473191 |  0.249559 |  0.306463 |  0.435258 |  0.692043 |                 168 |              0.422591 |              0.402922 |              0.325934 |            0.430085 |   0.740113 |        0.820817 |      0.865555 |            0.277542 |       0.950998 |                  0.0516732 |
| segformer_b0_rgb_384_smallmask        | jpeg_q50                   |      0.634578 |     0.464749 |  0.220284 |  0.30019  |  0.422736 |  0.689794 |                 176 |              0.407832 |              0.387133 |              0.309371 |            0.447034 |   0.739069 |        0.804329 |      0.851157 |            0.273305 |       0.950998 |                  0.0495111 |
| segformer_b0_rgb_384_smallmask        | gaussian_blur_light        |      0.650456 |     0.481982 |  0.271757 |  0.330741 |  0.432574 |  0.691835 |                 166 |              0.439784 |              0.42121  |              0.346315 |            0.349576 |   0.765051 |        0.847592 |      0.883424 |            0.385593 |       0.945554 |                  0.0439835 |
| segformer_b0_rgb_384_smallmask        | gaussian_blur_medium       |      0.65019  |     0.48169  |  0.260562 |  0.324067 |  0.424959 |  0.689755 |                 173 |              0.436054 |              0.417174 |              0.347138 |            0.317797 |   0.763797 |        0.842019 |      0.877802 |            0.387712 |       0.941924 |                  0.043888  |
| segformer_b0_rgb_384_smallmask        | gaussian_noise_light       |      0.63669  |     0.467018 |  0.256747 |  0.303563 |  0.431567 |  0.684152 |                 165 |              0.417671 |              0.396443 |              0.321285 |            0.427966 |   0.739007 |        0.817881 |      0.862238 |            0.283898 |       0.945554 |                  0.115246  |
| segformer_b0_rgb_384_smallmask        | gaussian_noise_medium      |      0.61747  |     0.446623 |  0.176673 |  0.24941  |  0.377617 |  0.675175 |                 201 |              0.362886 |              0.339207 |              0.269824 |            0.514831 |   0.729412 |        0.7627   |      0.814145 |            0.222458 |       0.956443 |                  0.113231  |
| segformer_b0_rgb_384_smallmask        | combined_jpeg70_blur_light |      0.645345 |     0.476391 |  0.245894 |  0.310044 |  0.42919  |  0.693204 |                 176 |              0.424155 |              0.402196 |              0.327073 |            0.40678  |   0.741059 |        0.822672 |      0.866434 |            0.302966 |       0.940109 |                  0.0514796 |
| efficientnetb0_unet_rgb_384_smallmask | clean_png                  |      0.576974 |     0.405456 |  0.283306 |  0.31737  |  0.412657 |  0.568471 |                 165 |              0.420648 |              0.401411 |              0.339853 |            0.239407 |   0.787781 |        0.861565 |      0.889    |            0.569915 |       0.889292 |                  0.0497742 |
| efficientnetb0_unet_rgb_384_smallmask | jpeg_q90                   |      0.571519 |     0.400088 |  0.268983 |  0.308664 |  0.398815 |  0.568249 |                 168 |              0.410354 |              0.390152 |              0.334596 |            0.258475 |   0.779283 |        0.85171  |      0.880973 |            0.544492 |       0.887477 |                  0.0614125 |
| efficientnetb0_unet_rgb_384_smallmask | jpeg_q70                   |      0.572602 |     0.401151 |  0.173218 |  0.290211 |  0.397081 |  0.576286 |                 194 |              0.392295 |              0.370466 |              0.310755 |            0.252119 |   0.762449 |        0.830166 |      0.862092 |            0.561441 |       0.84755  |                  0.0516374 |
| efficientnetb0_unet_rgb_384_smallmask | jpeg_q50                   |      0.567456 |     0.396118 |  0.104178 |  0.267102 |  0.375347 |  0.573345 |                 225 |              0.363992 |              0.347032 |              0.291585 |            0.252119 |   0.751227 |        0.805658 |      0.842077 |            0.550847 |       0.833031 |                  0.0515587 |
| efficientnetb0_unet_rgb_384_smallmask | gaussian_blur_light        |      0.579556 |     0.40801  |  0.264944 |  0.303039 |  0.409558 |  0.571095 |                 172 |              0.418589 |              0.402769 |              0.347396 |            0.224576 |   0.785366 |        0.85945  |      0.885346 |            0.584746 |       0.876588 |                  0.0454105 |
| efficientnetb0_unet_rgb_384_smallmask | gaussian_blur_medium       |      0.573435 |     0.401969 |  0.234768 |  0.293216 |  0.389345 |  0.569456 |                 187 |              0.411548 |              0.393421 |              0.343068 |            0.207627 |   0.775207 |        0.851197 |      0.877484 |            0.597458 |       0.85118  |                  0.0456426 |
| efficientnetb0_unet_rgb_384_smallmask | gaussian_noise_light       |      0.556857 |     0.385864 |  0.218593 |  0.277144 |  0.374625 |  0.562349 |                 194 |              0.385154 |              0.35899  |              0.306663 |            0.322034 |   0.764291 |        0.813179 |      0.84519  |            0.495763 |       0.885662 |                  0.120033  |
| efficientnetb0_unet_rgb_384_smallmask | gaussian_noise_medium      |      0.544164 |     0.373781 |  0.137063 |  0.220882 |  0.328661 |  0.550203 |                 234 |              0.339668 |              0.318552 |              0.27089  |            0.368644 |   0.733485 |        0.754756 |      0.78923  |            0.400424 |       0.876588 |                  0.121156  |
| efficientnetb0_unet_rgb_384_smallmask | combined_jpeg70_blur_light |      0.574875 |     0.403386 |  0.166507 |  0.287917 |  0.397414 |  0.573802 |                 195 |              0.393033 |              0.372659 |              0.320079 |            0.226695 |   0.773179 |        0.835961 |      0.865476 |            0.597458 |       0.84755  |                  0.0513155 |

## 11. JPEG sikistirma etkisi

JPEG90 hafif sikistirma, JPEG70 daha gercekci dagilim kaymasi, JPEG50 ise stres testi olarak yorumlanmalidir. Threshold yeniden secilmedigi icin bu bolum modelin dagilim kaymasi altindaki dogrudan davranisini gosterir.

## 12. Blur ve noise etkisi

Blur Q1 Dice'i dusuruyorsa kucuk sahtecilik bolgelerinin sinir/kenar sinyaline bagimliligi; noise authentic FP oranini artiriyorsa gurultu ile sahtecilik izinin karistigi raporlanmalidir.

## 13. Kucuk maske analizi

| mask_quartile   |   n |   mean_dice |   median_dice |   mean_iou |   median_iou |   dice_lt_005_count | model_name                            |
|:----------------|----:|------------:|--------------:|-----------:|-------------:|--------------------:|:--------------------------------------|
| Q1              | 138 |    0.276685 |      0.205295 |   0.195703 |     0.114429 |                  65 | segformer_b0_rgb_384_smallmask        |
| Q2              | 138 |    0.324935 |      0.335828 |   0.23864  |     0.201808 |                  53 | segformer_b0_rgb_384_smallmask        |
| Q3              | 137 |    0.439855 |      0.526625 |   0.329072 |     0.357428 |                  35 | segformer_b0_rgb_384_smallmask        |
| Q4              | 138 |    0.692177 |      0.736315 |   0.577018 |     0.582687 |                  10 | segformer_b0_rgb_384_smallmask        |
| Q1              | 138 |    0.283306 |      0.232015 |   0.200292 |     0.131251 |                  64 | efficientnetb0_unet_rgb_384_smallmask |
| Q2              | 138 |    0.31737  |      0.320427 |   0.230276 |     0.190853 |                  57 | efficientnetb0_unet_rgb_384_smallmask |
| Q3              | 137 |    0.412657 |      0.482133 |   0.298968 |     0.317638 |                  33 | efficientnetb0_unet_rgb_384_smallmask |
| Q4              | 138 |    0.568471 |      0.573666 |   0.431961 |     0.402197 |                  11 | efficientnetb0_unet_rgb_384_smallmask |

## 14. Failure case analizi

| model_name                            | failure_group            |   n | csv                                                                                                                              | png                                                                                                                              |
|:--------------------------------------|:-------------------------|----:|:---------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| segformer_b0_rgb_384_smallmask        | best_forged_cases        |  12 | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/best_forged_cases.csv               | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/best_forged_cases.png               |
| segformer_b0_rgb_384_smallmask        | worst_forged_cases       |  12 | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/worst_forged_cases.csv              | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/worst_forged_cases.png              |
| segformer_b0_rgb_384_smallmask        | small_mask_failures      |  12 | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/small_mask_failures.csv             | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/small_mask_failures.png             |
| segformer_b0_rgb_384_smallmask        | false_positive_authentic |  12 | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/false_positive_authentic.csv        | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/false_positive_authentic.png        |
| segformer_b0_rgb_384_smallmask        | false_negative_forged    |  12 | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/false_negative_forged.csv           | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/false_negative_forged.png           |
| segformer_b0_rgb_384_smallmask        | large_mask_success       |  12 | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/large_mask_success.csv              | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/large_mask_success.png              |
| segformer_b0_rgb_384_smallmask        | large_mask_failure       |  12 | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/large_mask_failure.csv              | /kaggle/working/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/large_mask_failure.png              |
| efficientnetb0_unet_rgb_384_smallmask | best_forged_cases        |  12 | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/best_forged_cases.csv        | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/best_forged_cases.png        |
| efficientnetb0_unet_rgb_384_smallmask | worst_forged_cases       |  12 | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/worst_forged_cases.csv       | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/worst_forged_cases.png       |
| efficientnetb0_unet_rgb_384_smallmask | small_mask_failures      |  12 | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/small_mask_failures.csv      | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/small_mask_failures.png      |
| efficientnetb0_unet_rgb_384_smallmask | false_positive_authentic |  12 | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/false_positive_authentic.csv | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/false_positive_authentic.png |
| efficientnetb0_unet_rgb_384_smallmask | false_negative_forged    |  12 | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/false_negative_forged.csv    | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/false_negative_forged.png    |
| efficientnetb0_unet_rgb_384_smallmask | large_mask_success       |  12 | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/large_mask_success.csv       | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/large_mask_success.png       |
| efficientnetb0_unet_rgb_384_smallmask | large_mask_failure       |  12 | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/large_mask_failure.csv       | /kaggle/working/experiments_full/final_analysis/efficientnetb0_unet_rgb_384_smallmask/failure_cases/large_mask_failure.png       |

## 15. Istatistiksel test sonuclari

| comparison                                                              | metric   | subset   |   n |   mean_a |    mean_b |   mean_diff |   paired_t_stat |   paired_t_p |   wilcoxon_stat |   wilcoxon_p |   cohens_d |   rank_biserial |
|:------------------------------------------------------------------------|:---------|:---------|----:|---------:|----------:|------------:|----------------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| segformer_b0_rgb_384_smallmask vs efficientnetb0_unet_rgb_384_smallmask | dice     | forged   | 551 | 0.433401 | 0.39542   |  0.0379816  |        4.56048  |  6.29641e-06 |           42565 |  8.2824e-05  |  0.194283  |       0.210882  |
| segformer_b0_rgb_384_smallmask vs efficientnetb0_unet_rgb_384_smallmask | iou      | forged   | 551 | 0.335119 | 0.290359  |  0.0447606  |        5.87441  |  7.35298e-09 |           40920 |  6.62806e-06 |  0.250258  |       0.241379  |
| segformer_b0_rgb_384_smallmask vs efficientnetb0_unet_rgb_384_smallmask | dice     | Q1       | 138 | 0.276685 | 0.283306  | -0.00662034 |       -0.463977 |  0.6434      |            2344 |  0.160975    | -0.0394964 |      -0.157592  |
| segformer_b0_rgb_384_smallmask vs efficientnetb0_unet_rgb_384_smallmask | dice     | Q2       | 138 | 0.324935 | 0.31737   |  0.00756511 |        0.507079 |  0.612915    |            2438 |  0.529187    |  0.0431654 |       0.0717685 |
| segformer_b0_rgb_384_smallmask vs segformer_b0_rgb_full                 | dice     | forged   | 551 | 0.433401 | 0.280098  |  0.153303   |       15.0138   |  5.96689e-43 |           31217 |  4.09699e-33 |  0.63961   |       0.589455  |
| segformer_b0_rgb_384_smallmask vs segformer_b0_rgb_full                 | iou      | forged   | 551 | 0.335119 | 0.211926  |  0.123193   |       15.5311   |  2.27224e-45 |           30981 |  1.90857e-33 |  0.661647  |       0.592559  |
| segformer_b0_rgb_384_smallmask vs segformer_b0_rgb_full                 | dice     | Q1       | 138 | 0.276685 | 0.0323276 |  0.244358   |        9.66633  |  3.58713e-17 |            2054 |  5.66051e-09 |  0.822852  |       0.571682  |
| segformer_b0_rgb_384_smallmask vs segformer_b0_rgb_full                 | dice     | Q2       | 138 | 0.324935 | 0.205271  |  0.119664   |        6.67738  |  5.57926e-10 |            2658 |  5.55083e-06 |  0.568416  |       0.44573   |
| efficientnetb0_unet_rgb_384_smallmask vs efficientnetb0_unet_rgb_full   | dice     | forged   | 551 | 0.39542  | 0.268308  |  0.127112   |       11.942    |  2.17171e-29 |           41576 |  3.04116e-20 |  0.508747  |       0.453221  |
| efficientnetb0_unet_rgb_384_smallmask vs efficientnetb0_unet_rgb_full   | iou      | forged   | 551 | 0.290359 | 0.194608  |  0.0957511  |       11.6843   |  2.50382e-28 |           41868 |  6.2802e-20  |  0.49777   |       0.449381  |
| efficientnetb0_unet_rgb_384_smallmask vs efficientnetb0_unet_rgb_full   | dice     | Q1       | 138 | 0.283306 | 0.0686349 |  0.214671   |        8.35827  |  6.3681e-14  |            2471 |  7.80251e-07 |  0.711503  |       0.484725  |
| efficientnetb0_unet_rgb_384_smallmask vs efficientnetb0_unet_rgb_full   | dice     | Q2       | 138 | 0.31737  | 0.212043  |  0.105327   |        5.69283  |  7.29318e-08 |            3249 |  0.00101345  |  0.484606  |       0.32249   |
| segformer_b0_rgb_384_smallmask vs unetpp_resnet34_rgb_full              | dice     | forged   | 551 | 0.433401 | 0.326815  |  0.106586   |       11.4211   |  2.94656e-27 |           38213 |  4.64465e-24 |  0.486554  |       0.497449  |
| segformer_b0_rgb_384_smallmask vs unetpp_resnet34_rgb_full              | iou      | forged   | 551 | 0.335119 | 0.234249  |  0.10087    |       12.3124   |  6.13061e-31 |           37178 |  2.64532e-25 |  0.524524  |       0.51106   |
| segformer_b0_rgb_384_smallmask vs unetpp_resnet34_rgb_full              | dice     | Q1       | 138 | 0.276685 | 0.163525  |  0.11316    |        5.81172  |  4.13928e-08 |            2782 |  1.87501e-05 |  0.494726  |       0.419873  |
| segformer_b0_rgb_384_smallmask vs unetpp_resnet34_rgb_full              | dice     | Q2       | 138 | 0.324935 | 0.267996  |  0.0569397  |        3.16091  |  0.00193635  |            3835 |  0.0412167   |  0.269075  |       0.200292  |
| efficientnetb0_unet_rgb_384_smallmask vs unetpp_resnet34_rgb_full       | dice     | forged   | 551 | 0.39542  | 0.326815  |  0.0686048  |        7.22319  |  1.70159e-12 |           51070 |  2.41974e-11 |  0.307718  |       0.328362  |
| efficientnetb0_unet_rgb_384_smallmask vs unetpp_resnet34_rgb_full       | iou      | forged   | 551 | 0.290359 | 0.234249  |  0.0561092  |        7.32314  |  8.6618e-13  |           50333 |  6.187e-12   |  0.311976  |       0.338055  |
| efficientnetb0_unet_rgb_384_smallmask vs unetpp_resnet34_rgb_full       | dice     | Q1       | 138 | 0.283306 | 0.163525  |  0.11978    |        6.02362  |  1.48461e-08 |            2506 |  1.13961e-06 |  0.512765  |       0.477427  |
| efficientnetb0_unet_rgb_384_smallmask vs unetpp_resnet34_rgb_full       | dice     | Q2       | 138 | 0.31737  | 0.267996  |  0.0493746  |        2.80784  |  0.00571619  |            4174 |  0.186545    |  0.239019  |       0.129601  |

## 16. Final pipeline onerisi

Aday 1: `SegFormer-B0 384 balanced` - `Localization-oriented final model`. Forged bolgenin en iyi lokalizasyonu istendiginde kullanilmalidir.

Aday 2: `EfficientNetB0-UNet 384 balanced / low false alarm` - `Conservative low-false-alarm final model`. Gercek goruntulerde yanlis alarm maliyeti yuksek oldugunda kullanilmalidir.

Eger ana hedef piksel/bilesen lokalizasyon basarisiysa final model SegFormer-B0 384'tur. Eger pratik kullanimda dusuk yanlis alarm ve goruntu duzeyi guvenilirlik oncelikliyse final model EfficientNetB0-UNet 384'tur. Tezde iki model birlikte raporlanmalidir; cunku farkli operasyonel onceliklere hizmet etmektedirler.

## 17. Sinirliliklar

Robustness bolumu checkpoint gerektirir. Checkpoint eksikse clean analiz cached probability map ile tamamlanir, ancak degrade goruntu forward pass'i yapilamaz. Sonuclar validation secimli threshold'lara baglidir ve test setinde yeniden ayar yapilmaz.

## 18. Gelecek calisma onerileri

DINOv2-lite limited unfreeze, daha guclu domain augmentation, JPEG/blur/noise ile validation-time threshold adaptation ve uncertainty-aware post-processing gelecek calisma olarak degerlendirilebilir.
