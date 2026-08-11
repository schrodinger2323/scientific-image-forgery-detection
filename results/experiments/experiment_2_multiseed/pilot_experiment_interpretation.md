# Pilot Experiment Interpretation

This interpretation is based on the completed three-seed pilot stability experiment. The same 300 authentic + 300 forged image subset from Experiment 1 was reused, and only seed-dependent split/training randomness changed.

## Setup

- Seeds: 42, 123, 2025
- Models: plain_unet, unetplusplus, efficientnetb0_unet, deeplabv3plus, segformer_b0
- Split: 60/20/20 stratified group-aware split with `image_id` / `group_id` as the group key
- Fixed subset check: each Experiment 2 seed cache uses the same `sample_id` set as Experiment 1
- Primary localization metric: forged-only pixel F1

For seed 42, the train, val-tune, and internal-test split files are identical to Experiment 1. This makes seed 42 a strict rerun point under the current code revision.

## Stability

The most stable and strongest model is `segformer_b0`. It has the best mean forged-only pixel F1 and image-level metrics:

| model_name | forged_pixel_f1 | forged_pixel_iou | forged_pixel_precision | forged_pixel_recall | image_f1 | image_roc_auc | average_epochs_ran |
| --- | --- | --- | --- | --- | --- | --- | --- |
| segformer_b0 | 0.2666 +/- 0.0585 | 0.1815 +/- 0.0426 | 0.5409 +/- 0.0336 | 0.2024 +/- 0.0523 | 0.6777 +/- 0.0366 | 0.7375 +/- 0.0342 | 37.33 |
| efficientnetb0_unet | 0.1866 +/- 0.0077 | 0.1272 +/- 0.0097 | 0.3038 +/- 0.1311 | 0.1840 +/- 0.0521 | 0.5608 +/- 0.0745 | 0.5991 +/- 0.0930 | 16.33 |
| unetplusplus | 0.0528 +/- 0.0311 | 0.0329 +/- 0.0209 | 0.1233 +/- 0.0699 | 0.0462 +/- 0.0175 | 0.3367 +/- 0.0309 | 0.5298 +/- 0.0534 | 22.67 |
| deeplabv3plus | 0.0283 +/- 0.0114 | 0.0174 +/- 0.0075 | 0.1022 +/- 0.0692 | 0.0208 +/- 0.0109 | 0.2402 +/- 0.1399 | 0.5549 +/- 0.0290 | 30.00 |
| plain_unet | 0.0255 +/- 0.0343 | 0.0149 +/- 0.0198 | 0.0247 +/- 0.0352 | 0.0572 +/- 0.0837 | 0.2195 +/- 0.2164 | 0.5182 +/- 0.0635 | 9.67 |

Using coefficient of variation on forged-only pixel F1 as a practical stability signal, `efficientnetb0_unet` is the most stable model numerically, because its forged F1 varies very little across seeds. `segformer_b0` is also stable enough and clearly stronger in absolute performance. `plain_unet`, `unetplusplus`, and `deeplabv3plus` are not stable enough to be primary candidates in this pilot stage.

## SegFormer-B0 And DeepLabV3+

The superiority of `segformer_b0` is preserved under seed changes. It ranks first for forged-only pixel F1 in all three seeds and also has the best mean image F1 and ROC-AUC.

The superiority of `deeplabv3plus` from Experiment 1 is not preserved in Experiment 2. In Experiment 1, DeepLabV3+ was close to SegFormer-B0 on forged-only localization. In the three-seed rerun, however, DeepLabV3+ drops behind SegFormer-B0 and EfficientNetB0 U-Net by a large margin. Therefore, DeepLabV3+ should not be treated as a stable main candidate based on the current Experiment 2 evidence.

One technical caveat is that the Experiment 2 DeepLabV3+ run uses the corrected internal lightweight DeepLabV3+ fallback after the ASPP image-pooling BatchNorm issue was fixed. For strict comparisons, the Experiment 2 seed-42 DeepLabV3+ result should be used as the current-code reference instead of mixing old and new DeepLabV3+ artifacts.

## EfficientNetB0 U-Net

EfficientNetB0 U-Net is the second strongest and most consistent model in Experiment 2. Its forged-only pixel F1 is 0.1866 +/- 0.0077, which indicates strong seed stability.

It behaves more conservatively than SegFormer-B0 in the sense that it tends to produce lower recall and lower image-level detection strength, while keeping a moderate forged precision. Across the completed runs, its forged precision is 0.3038 +/- 0.1311 and forged recall is 0.1840 +/- 0.0521. This makes it a useful conservative CNN baseline, but not the strongest final candidate.

## ResNet50 U-Net

ResNet50 U-Net was not included in Experiment 2 because Experiment 1 already showed an undesirable behavior pattern. Its forged-only pixel F1 was 0.1047 and forged-only IoU was 0.0592, both below SegFormer-B0, DeepLabV3+, and EfficientNetB0 U-Net in the first pilot.

More importantly, ResNet50 U-Net reached this by predicting almost everything as positive: forged recall was very high at 0.9178, but forged precision was only 0.0597, authentic positive rate was 1.0, and the selected threshold was 0.10. This indicates severe over-detection rather than useful localization. For that reason, it was not selected as a main candidate for the seed-stability stage.

## Why Forged-Only Pixel F1 Is Primary

All-image pixel metrics are misleading in this task because authentic images have all-zero masks. A model can obtain high all-image pixel scores by predicting little manipulated area on authentic images while still failing to localize forged regions.

Forged-only pixel F1 evaluates only the images where manipulated pixels actually exist. It balances precision and recall on the forged subset, so it directly answers the main segmentation question: can the model localize forged regions when forgery is present? Forged-only pixel IoU is used as a secondary localization metric, while image F1 and ROC-AUC summarize image-level detection behavior.

## Practical Conclusion

For the next report and follow-up experiments, `segformer_b0` should be treated as the primary candidate. `efficientnetb0_unet` should be kept as the strongest stable CNN baseline. `deeplabv3plus` can be discussed as a model that looked promising in the first pilot but did not remain stable in the repeated pilot setting. `plain_unet` and `unetplusplus` remain useful baselines, but their localization results are too weak for main-model status.
