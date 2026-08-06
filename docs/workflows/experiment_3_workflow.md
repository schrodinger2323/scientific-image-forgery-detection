# Deney 3 Adim Adim Islem Akisi

Bu not, `analysis_review/experiment_3_codex_full_analysis.md`,
`experiments/experiment_comparison.csv` ve `experiments/*` altindaki
sonuc artefaktlari okunarak hazirlanmistir. Amac Deney 3'te RGB, SRM ve
edge-aware multitask yaklasimlarinin nasil uygulandigini adim adim
aciklamaktir.

## 1. Deneyin temel amaci

Deney 3, U-Net++ ailesi uzerinde input temsili ve edge-aware multitask
tasariminin etkisini olcmek icin yapildi.

Ana soru:

```text
RGB baseline'a SRM forensic residual kanallari ve edge-aware multitask
hedef eklemek lokalizasyonu iyilestiriyor mu?
```

Bu deney model ailesini cok genisletmez; U-Net++ uzerinden temsil/loss/inference
ablation'i yapar.

## 2. Calistirilan konfigurasyonlar

Deney 3'te uc ana konfigurasyon vardir:

| Konfigurasyon | Input | Encoder | Output / inference |
|---|---|---|---|
| `unetpp_resnet34_rgb_baseline` | RGB | ResNet34 | surface mask |
| `unetpp_resnet34_rgb_srm_edge_multitask` | RGB + SRM | ResNet34 | edge-enhanced |
| `unetpp_resnet50_rgb_srm_edge_multitask` | RGB + SRM | ResNet50 | edge-enhanced |

RGB baseline yalnizca 3-kanal goruntu kullanir. SRM modelleri RGB'ye ek olarak
5 high-pass residual kanali uretip girise ekler.

## 3. Shared split dogrulamasi

Deney 3'te karsilastirmanin adil olmasi icin ayni shared split kullanildi.

Split boyutlari:

| Split | Toplam | Authentic | Forged |
|---|---:|---:|---:|
| Full | 5128 | 2377 | 2751 |
| Train | 3590 | 1665 | 1925 |
| Validation | 515 | 240 | 275 |
| Test | 1023 | 472 | 551 |

`experiments/_shared_splits_seed42` ile ucuncu deney klasorundeki split
kopyasi SHA256 ve satir-satir karsilastirma ile ayni bulundu. Leakage kontrolu
train-val, train-test ve val-test icin 0 olarak raporlandi.

## 4. Veri ve maske hazirlama

Temel veri akisi:

```text
PNG image oku
-> forged icin .npy mask oku
-> authentic icin sifir mask uret
-> maskeyi binary hale getir
-> image/mask 256x256 resize
-> train augmentasyonu uygula
-> validation/test deterministik kalsin
```

Maskelerde cok kanalli yapi varsa:

```text
np.any(mask > 0, axis=kanal_ekseni)
```

ile tek binary maskeye indirildi.

## 5. RGB ve SRM temsili

RGB baseline:

```text
input = [R, G, B]
```

SRM modelleri:

```text
RGB image
-> 5 adet normalize high-pass SRM residual kanali hesapla
-> input = [R, G, B, SRM1, SRM2, SRM3, SRM4, SRM5]
```

Bu nedenle SRM modellerinin input'u 8 kanallidir. Buradaki amac, kopyalama,
ekleme veya manipulasyon izlerinin renk uzayindan cok noise/residual uzayinda
daha belirgin olup olmadigini test etmektir.

## 6. Edge target nasil uretildi?

Edge-aware modellerde ikinci bir egitim hedefi vardir. Edge target, ground-truth
maskeden turetildi:

```text
edge_target = dilate(mask) - erode(mask)
```

Yani edge hedefi manuel annotation degildir; GT maskenin morfolojik sinir
bandindan uretilmistir.

## 7. Loss ve multitask egitim

RGB baseline tek maske cikisi uretir:

```text
loss = mask_loss
```

SRM + edge multitask modeller iki cikis uretir:

```text
surface mask probability
edge probability
```

Multitask loss:

```text
total_loss = mask_loss + 0.4 * edge_loss
```

Burada edge loss, modeli sinir bolgelerine daha dikkatli hale getirmek icin
egitim sirasinda kullanildi.

## 8. Surface ve edge-enhanced inference

`surface` modu:

```text
p_final = p_surface
binary_mask = p_final >= threshold
```

`edge_enhanced` modu:

```text
p_final = p_surface * (1 + 0.2 * p_edge)
binary_mask = p_final >= threshold
```

Yani edge-enhanced sadece rapor terimi degildir; model egitiminde ikinci hedef
vardir ve inference sirasinda edge olasiligi surface olasiligini guclendirir.

## 9. Ortak egitim ayarlari

| Ayar | Deger |
|---|---|
| Seed | 42 |
| Image size | 256 |
| Batch size | 8 |
| Epoch ust siniri | 40 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Early stopping patience | 8 |
| Scheduler | ReduceLROnPlateau |
| Edge loss weight | 0.4 |
| Edge enhance weight | 0.2 |
| Robustness | kapali |

Best checkpoint validation Dice'a gore secildi.

## 10. Threshold secimi

Threshold test setinden secilmedi.

```text
validation probability map
-> 0.10-0.90 threshold sweep
-> en iyi validation threshold
-> test setine sabit uygulama
```

Uc ana konfigurasyonun secilen threshold'u:

| Model | Selected threshold |
|---|---:|
| RGB ResNet34 baseline | 0.90 |
| RGB+SRM ResNet34 edge multitask | 0.90 |
| RGB+SRM ResNet50 edge multitask | 0.90 |

## 11. Ana test sonuclari

`experiments/experiment_comparison.csv` sonuclarina gore:

| Konfigurasyon | Test Dice | Test IoU | Precision | Recall | AUPRC | Image F1 | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB ResNet34 baseline | 0.5259 | 0.3567 | 0.4304 | 0.6757 | 0.5807 | 0.7411 | 0.1995 |
| RGB+SRM ResNet34 edge | 0.4460 | 0.2870 | 0.3281 | 0.6965 | 0.3448 | 0.7114 | 0.2793 |
| RGB+SRM ResNet50 edge | 0.4525 | 0.2924 | 0.3735 | 0.5738 | 0.3563 | 0.7028 | 0.2703 |

Ana Dice/IoU siralamasi:

```text
1. RGB ResNet34 baseline
2. RGB+SRM ResNet50 edge multitask
3. RGB+SRM ResNet34 edge multitask
```

Boundary F1 siralamasi:

```text
1. RGB+SRM ResNet34 edge multitask
2. RGB+SRM ResNet50 edge multitask
3. RGB ResNet34 baseline
```

## 12. Surface vs edge-enhanced gozlemi

Multitask modellerde surface-only test performansi edge-enhanced moda gore
daha yuksek Dice verdi:

| Model | Surface Dice | Edge-enhanced Dice |
|---|---:|---:|
| ResNet34 RGB+SRM multitask | 0.4974 | 0.4460 |
| ResNet50 RGB+SRM multitask | 0.4663 | 0.4525 |

Bu bulgu onemlidir:

```text
edge-aware yapi boundary F1'i iyilestirdi,
fakat pixel overlap/Dice tarafinda RGB baseline'i gecemedi.
```

## 13. Deney 4'e gecis gerekcesi

Deney 3 sonucunda RGB baseline'in ana Dice/IoU tarafinda daha guclu oldugu
goruldu. Bu nedenle Deney 4'te SRM/edge ekleri yerine RGB input ile daha
guclu model ailelerinin full-data karsilastirmasina gecildi.

Deney 3'ten cikan tez yorumu:

```text
SRM + edge-aware multitask boundary duyarliligini artirdi,
ancak genel mask overlap metriklerinde RGB U-Net++ baseline'i gecemedi.
```

## 14. Kaynak dosyalar

```text
experiments/experiment_comparison.csv
experiments/_shared_splits_seed42/
experiments/unetpp_resnet34_rgb_baseline/
experiments/unetpp_resnet34_rgb_srm_edge_multitask/
experiments/unetpp_resnet50_rgb_srm_edge_multitask/
analysis_review/experiment_3_codex_full_analysis.md
deney3_rapor.md
analysis_review/deney3_rapor.md
```
