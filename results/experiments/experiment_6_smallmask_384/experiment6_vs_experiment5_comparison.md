# Deney 6 vs Deney 5 Karsilastirmasi

Deney 6 basari kosulu: alti kriterden en az ikisinin saglanmasi.

| experiment6_model                     | experiment6_strategy    |   delta_Q1 Dice |   delta_Q2 Dice |   delta_forged_dice |   delta_component F1 @0.10 |   exp6_authentic FP rate |   success_criteria_met | success   |
|:--------------------------------------|:------------------------|----------------:|----------------:|--------------------:|---------------------------:|-------------------------:|-----------------------:|:----------|
| efficientnetb0_unet_rgb_384_smallmask | best_forged_dice        |             nan |             nan |          0.0555858  |                 0.0782433  |                 0.279661 |                      3 | True      |
| efficientnetb0_unet_rgb_384_smallmask | best_small_mask_q1_dice |             nan |             nan |          0.037047   |                -0.0567602  |                 0.472458 |                      1 | False     |
| efficientnetb0_unet_rgb_384_smallmask | balanced_final_score    |             nan |             nan |          0.0570924  |                 0.0955007  |                 0.239407 |                      3 | True      |
| efficientnetb0_unet_rgb_384_smallmask | low_false_alarm         |             nan |             nan |          0.0704997  |                 0.100595   |                 0.239407 |                      3 | True      |
| efficientnetb0_unet_rgb_384_smallmask | small_object_practical  |             nan |             nan |          0.0558115  |                 0.0736278  |                 0.300847 |                      3 | True      |
| segformer_b0_rgb_384_smallmask        | best_forged_dice        |             nan |             nan |          0.0866817  |                 0.0671721  |                 0.400424 |                      2 | True      |
| segformer_b0_rgb_384_smallmask        | best_small_mask_q1_dice |             nan |             nan |          0.00241905 |                -0.00269127 |                 0.631356 |                      2 | True      |
| segformer_b0_rgb_384_smallmask        | balanced_final_score    |             nan |             nan |          0.0878038  |                 0.112138   |                 0.355932 |                      2 | True      |
| segformer_b0_rgb_384_smallmask        | low_false_alarm         |             nan |             nan |          0.0607338  |                 0.0667552  |                 0.290254 |                      3 | True      |
| segformer_b0_rgb_384_smallmask        | small_object_practical  |             nan |             nan |          0.0878104  |                 0.0916674  |                 0.400424 |                      2 | True      |
