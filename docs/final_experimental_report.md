# ReCodAI-LUC Bilimsel Görüntü Sahtekârlığı Tespiti: Deneysel Süreç ve Final Model Analizi

---

## İçindekiler

1. [Giriş](#1-giriş)
2. [Veri Seti ve Paylaşılan Deneysel Protokol](#2-veri-seti-ve-paylaşılan-deneysel-protokol)
3. [Deney 1 – 15-Model Mimari Tarama (Küçük Alt Küme Pilot)](#3-deney-1--15-model-mimari-tarama-küçük-alt-küme-pilot)
4. [Deney 2 – Üç Tohum Stabilite Analizi](#4-deney-2--üç-tohum-stabilite-analizi)
5. [Deney 3 – UNet++ Tam Veri Kümesi Analizi](#5-deney-3--unet-tam-veri-kümesi-analizi)
6. [Deney 4 – Dört Mimari Ailesi Karşılaştırması (Tam Veri Kümesi)](#6-deney-4--dört-mimari-ailesi-karşılaştırması-tam-veri-kümesi)
7. [Deney 5 – Kalibrasyon ve Post-processing Optimizasyonu](#7-deney-5--kalibrasyon-ve-post-processing-optimizasyonu)
8. [Deney 6 – 384×384 Çözünürlük ve Küçük Maske Lokalizasyonu](#8-deney-6--384384-çözünürlük-ve-küçük-maske-lokalizasyonu)
9. [Final Analiz – İki Final Model Kapsamlı Değerlendirmesi](#9-final-analiz--iki-final-model-kapsamlı-değerlendirmesi)
10. [Sonuç ve Tartışma](#10-sonuç-ve-tartışma)
11. [Sınırlılıklar ve Gelecek Çalışma Önerileri](#11-sınırlılıklar-ve-gelecek-çalışma-önerileri)

---

## 1. Giriş

Bilimsel görüntü manipülasyonu, akademik araştırmaların güvenilirliğini tehdit eden ciddi bir bütünlük sorunudur. Yayımlanan makalelerdeki görüntülerin —mikrograf, blot, floresan görüntüsü veya veri grafiği— kopyalanıp yapıştırılması, yansıtılması ya da başka türlü manipüle edilmesi hem tekrarlanamazlık krizini derinleştirmekte hem de araştırma etiği ihlallerinin tespit maliyetini artırmaktadır. Manuel inceleme iş gücü yoğundur ve uzman editörlerin bile kaçırabileceği ince manipülasyonlara karşı tutarlı bir performans sergileyemez. Otomatik tespit sistemleri bu boşluğu doldurmayı amaçlar; ancak bilimsel görüntüler doğal fotoğraflardan farklı bir görsel yapıya sahip olduğundan genel amaçlı sahtekârlık tespit yaklaşımları bu alana doğrudan aktarılamaz.

Bu çalışma, ReCodAI-LUC Scientific Image Forgery Detection veri kümesi üzerinde gerçekleştirilen altı aşamalı deneysel bir sürecin ve ardından yürütülen kapsamlı bir final analizinin raporudur. Çalışmanın temel hedefi, sahte bölgeyi piksel düzeyinde lokalize edebilen ve aynı zamanda gerçek görüntülerde güvenilir biçimde alarm üretmeyen bir segmentasyon modeli geliştirmektir.

Deneysel yol, giderek daralan ve derinleşen bir odak çizgisini izlemiştir. İlk iki deney, küçük bir alt küme üzerinde 15 farklı mimari ailesiyle geniş bir tarama gerçekleştirerek güçlü adayları belirledi. Üçüncü deney, bir adayı tam veri kümesinde derinlemesine inceledi. Dördüncü deney, dört farklı mimari ailesini aynı koşullar altında karşılaştırdı. Beşinci deney, tespit edilen yüksek yanlış alarm sorununu post-processing optimizasyonuyla ele aldı. Altıncı deney, küçük sahtecilik bölgesi lokalizasyonu için çözünürlük artışının etkisini ölçtü. Final analiz ise seçilen iki model adayını bağımsız bir test seti üzerinde yeni bir eğitim yapmadan kapsamlı biçimde değerlendirdi.

---

## 2. Veri Seti ve Paylaşılan Deneysel Protokol

### 2.1 Veri Kümesi

Çalışmanın tamamı ReCodAI-LUC Scientific Image Forgery Detection veri kümesi üzerinde yürütülmüştür. Veri kümesi 5.128 görüntü içermektedir: 2.377 gerçek (authentic) ve 2.751 sahte (forged).

Sahte görüntüler, manipüle edilmiş bölgenin konumunu piksel düzeyinde gösteren ikili ground-truth maskelere sahiptir. Maskeler `.npy` formatında saklanmıştır; birden fazla maske kanalı varsa kanallar binary union (maksimum) işlemiyle tek bir ikili maskeye indirgenir. Gerçek görüntüler için sıfır maskesi oluşturulur.

### 2.2 Veri Bölmesi ve Leakage Kontrolü

Deney 3'ten itibaren çalışmanın tamamında aynı paylaşılan bölme kullanılmıştır. Bölme, görüntü kimliği (`image_id`) anahtarıyla katmanlı grup stratejisiyle ve rastgele tohum 42 sabitlenmiş olarak oluşturulmuştur. Bölmenin özdeşliği Deney 3'te SHA-256 hash ve satır bazlı karşılaştırmayla doğrulanmıştır.

**Tablo 1. Veri bölmesi — görüntü ve etiket dağılımı**

| Bölme | Görüntü Sayısı | Gerçek | Sahte |
|---|---:|---:|---:|
| Eğitim (%70) | 3.590 | 1.665 | 1.925 |
| Doğrulama (%10) | 515 | 240 | 275 |
| Test (%20) | 1.023 | 472 | 551 |
| **Toplam** | **5.128** | **2.377** | **2.751** |

Leakage kontrolü, üç bölme çifti için ayrı ayrı yapılmıştır:

| Kontrol | Ortak image_id |
|---|---:|
| Eğitim – Doğrulama | 0 |
| Eğitim – Test | 0 |
| Doğrulama – Test | 0 |

Bu kontrolün sıfır sonuç vermesi, hiçbir görüntünün birden fazla bölmede yer almadığını ve sonuçların iyimser sapma içermediğini göstermektedir.

### 2.3 Küçük Maske Quartile Tanımı

Sahte görüntüler, ground-truth maske piksel alanına göre dört eşit gruba (Q1–Q4) ayrılmıştır. Bu ayrım, küçük bölgelerdeki model performansını büyük bölgelerden bağımsız olarak değerlendirebilmek için tasarlanmıştır. Her quartile sınırı bölme bazında ayrı hesaplanmıştır; bu sayede sınır değerleri bir bölmenin alan dağılımına özgü kalır.

**Tablo 2. Test setinde maske büyüklüğü quartile sınırları**

| Quartile | n | Min Alan (px) | Max Alan (px) | Ort. Alan (px) | Ort. Alan Oranı |
|---|---:|---:|---:|---:|---:|
| Q1 (en küçük) | 138 | 158 | 2.488 | 1.054,6 | %1,11 |
| Q2 | 138 | 2.498 | 8.387 | 4.906,8 | %3,36 |
| Q3 | 137 | 8.484 | 30.018 | 15.777,7 | %7,49 |
| Q4 (en büyük) | 138 | 30.187 | 925.845 | 158.824,3 | %10,07 |

Q1 grubundaki görüntülerde sahtecilik bölgesi görüntünün ortalama yalnızca %1,11'ini kaplamaktadır. Bu son derece küçük sinyal, modellerin bu grupta zorlanmasının temel yapısal nedenidir.

### 2.4 Ortak Eğitim Protokolu

Deney 4'ten itibaren tüm modeller için ortak bir eğitim protokolü benimsenmiştir:

- **Optimizer:** AdamW, öğrenme hızı 1×10⁻⁴, ağırlık bozunumu 1×10⁻⁴
- **Batch boyutu:** 8
- **Maksimum epoch:** 40; erken durdurma sabır değeri 8
- **LR scheduler:** ReduceLROnPlateau (faktör 0,5, sabır 3)
- **Loss:** 0,5 × BCEWithLogitsLoss + 0,5 × DiceLoss
- **Augmentasyon:** Albumentations — yatay/dikey çevirme, 90° döndürme, kaydırma-ölçek-döndürme, parlaklık/kontrast, Gaussian bulanıklık, Gaussian gürültü, görüntü sıkıştırma; doğrulama/test için yalnızca yeniden boyutlandırma ve normalizasyon
- **Threshold seçimi:** Yalnızca doğrulama seti üzerinde; test setine sabit uygulanır

---

## 3. Deney 1 – 15-Model Mimari Tarama (Küçük Alt Küme Pilot)

### 3.1 Amaç ve Tasarım

Deneysel sürecin ilk adımı, geniş bir mimari uzayda hızlı bir ön eleme gerçekleştirmektir. Tam veri kümesinde 15 model eğitmek hem hesaplama maliyeti hem de zaman açısından elverişli değildir. Bu nedenle 300 gerçek + 300 sahte görüntüden oluşan dengeli bir alt küme seçilmiş ve bu alt küme üzerinde 15 farklı model ailesi karşılaştırılmıştır.

Eğitim ve değerlendirme protokolü şu parametrelerle sabitlenmiştir: görüntü boyutu 256×256, batch boyutu 8, maksimum epoch 40, öğrenme hızı 1×10⁻⁴, AdamW optimizer, tohum 42. Bölme oranları %60 eğitim / %20 doğrulama / %20 test şeklindedir.

### 3.2 Test Edilen Modeller

15 model, dört geniş kategoriye ayrılabilir:

**Genel amaçlı segmentasyon:** Plain UNet, EfficientNetB0-UNet, ResNet50-UNet, SegFormer-B0, DeepLabv3+, DINOv2-lite

**Kopyala-yapıştır manipülasyon tespitine özgü:** DoAGAN, Siamese CMFD, QDL-CMFD, Self-Correlated CMFD

**Derin öğrenme tabanlı adli araçlar:** BusterNet, CMFDFormer, MantraNet, MVSSNet, MVSSNet++

### 3.3 Sonuçlar

**Tablo 3. 15-model pilot karşılaştırması — Forged-only Pixel F1'e göre sıralanmış**

| Model | Forged F1 | Forged IoU | Forged Prec. | Forged Rec. | Görüntü F1 | Görüntü ROC-AUC | Threshold | Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | 0,2529 | 0,1712 | 0,5258 | 0,1852 | 0,6825 | 0,7383 | 0,90 | 40 |
| DeepLabv3+ | 0,2499 | 0,1661 | 0,4909 | 0,1821 | 0,6818 | 0,7397 | 0,90 | 37 |
| EfficientNetB0-UNet | 0,1783 | 0,1188 | 0,4537 | 0,1305 | 0,6364 | 0,6800 | 0,90 | 18 |
| ResNet50-UNet | 0,1047 | 0,0592 | 0,0597 | 0,9178 | 0,6667 | 0,4585 | 0,10 | 9 |
| DoAGAN | 0,0532 | 0,0353 | 0,0536 | 0,0847 | 0,2532 | 0,4825 | 0,90 | 9 |
| Siamese CMFD | 0,0454 | 0,0296 | 0,0524 | 0,0783 | 0,2750 | 0,4375 | 0,90 | 9 |
| BusterNet | 0,0434 | 0,0290 | 0,0588 | 0,0771 | 0,1892 | 0,4913 | 0,80 | 11 |
| QDL-CMFD | 0,0186 | 0,0103 | 0,1164 | 0,0117 | 0,2750 | 0,5150 | 0,90 | 40 |
| CMFDFormer | 0,0137 | 0,0086 | 0,0194 | 0,0135 | 0,1867 | 0,5253 | 0,90 | 9 |
| Self-Corr CMFD | 0,0133 | 0,0077 | 0,0199 | 0,0105 | 0,1176 | 0,4617 | 0,90 | 10 |
| Plain UNet | 0,0121 | 0,0074 | 0,0091 | 0,0184 | 0,0938 | 0,4475 | 0,75 | 10 |
| MVSSNet | 0,0119 | 0,0069 | 0,0258 | 0,0098 | 0,1471 | 0,4803 | 0,90 | 9 |
| DINOv2-lite | 0,0088 | 0,0046 | 0,1580 | 0,0047 | 0,4045 | 0,5958 | 0,90 | 28 |
| MantraNet | 0,0010 | 0,0005 | 0,0500 | 0,0005 | 0,1096 | 0,4581 | 0,90 | 40 |
| MVSSNet++ | 0,0007 | 0,0004 | 0,0167 | 0,0004 | 0,0312 | 0,4757 | 0,90 | 40 |

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Bu noktaya model_comparison_enriched.md verisini bar grafik olarak koyabilirsiniz: yatay eksende modeller, dikey eksende Forged Pixel F1. İlk üç modeli farklı renkte vurgulayın.

### 3.4 Yorum

Tabloda net bir iki kümeli ayrışma görülmektedir. SegFormer-B0 (0,2529), DeepLabv3+ (0,2499) ve EfficientNetB0-UNet (0,1783) diğer 12 modeli açık farkla geride bırakmıştır. Bu üç model, sahte bölgeyi belirli bir alan hassasiyetiyle lokalize edebildiğini gösteren anlamlı Forged F1 değerlerine ulaşmıştır.

Kopyala-yapıştır manipülasyon tespitine özgü modeller (DoAGAN, Siamese CMFD, QDL-CMFD, Self-Corr CMFD) ve özelleşmiş adli araçlar (BusterNet, MantraNet, MVSSNet, MVSSNet++) bu görevde kayda değer performans gösterememiştir. Bu sonuç beklenen bir bulgudur: bu modeller genellikle kopyalanan bölgelerdeki kendi kendine tutarlılık bozulmalarını (self-correlation artifacts) arar; bilimsel görüntülerdeki düzenleme türü ise çok daha geniş bir manipülasyon yelpazesini kapsar ve farklı görsel izler bırakır.

DINOv2-lite bu aşamada 28 epoch sonunda yalnızca 0,0088 Forged F1 üretti; ancak görüntü düzeyi ROC-AUC (0,5958) diğer düşük performanslı modellerin üzerindedir. Bu, dondurulmuş büyük ölçekli öz-denetimli öğrenme omurgasının küçük bir alt kümede ince ayar olmaksızın anlamlı lokalizasyon sinyali üretemediğine, ancak görüntü düzeyi ayrıştırma kapasitesinin mevcut olduğuna işaret etmektedir.

---

## 4. Deney 2 – Üç Tohum Stabilite Analizi

### 4.1 Amaç ve Tasarım

Pilot çalışmanın bir eksiği, tek bir rastgele tohumla (seed 42) elde edilen sonuçların model sıralama kararlılığını gösterip göstermediğinin bilinmemesidir. Bir modelin pilot üstünlüğü, başka bir tohum altında bozulabilir. Deney 2, bu riski kontrol etmek için Deney 1'in kısaltılmış aday listesini üç farklı tohum (42, 123, 2025) altında çalıştırmıştır.

Değerlendirilen beş model: Plain UNet, UNet++ (ResNet34), EfficientNetB0-UNet, DeepLabv3+, SegFormer-B0. Aynı 300 gerçek + 300 sahte alt küme ve %60/%20/%20 bölme oranı kullanılmıştır. Tohum değişimi; bölme atamasını, model başlatmasını ve veri yükleyici karıştırmasını etkilemiş; görüntü alt kümesi sabit tutulmuştur.

### 4.2 Sonuçlar ve Yorum

Üç tohum genelinde SegFormer-B0 ve EfficientNetB0-UNet, Forged Pixel F1 sıralamasında tutarlı biçimde öne çıkmıştır. Plain UNet, tüm tohum koşullarında düşük performans göstermiş; DeepLabv3+ rekabetçi bir profil sergilemiş ancak SegFormer'ın altında kalmıştır.

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Üç tohum için model performansını gosteren kutu grafik (box plot) veya çizgi grafik ekleyin: yatay eksen model adları, dikey eksen Forged F1, üç nokta her tohumu temsil etsin. Kaynak: `deney_2/pilot_seed_comparison.csv`

Bu stabilite analizi iki temel kararı desteklemiştir: (a) SegFormer-B0 ve EfficientNetB0-UNet güvenilir adaylar olarak tam veri kümesine taşınacaktır; (b) Plain UNet ve alt performanslı alan-spesifik modeller elenmiştir.

---

## 5. Deney 3 – UNet++ Tam Veri Kümesi Analizi

### 5.1 Amaç ve Tasarım

Pilot deneylerin ardından UNet++ ailesi, farklı encoder ve giriş modalitesi kombinasyonlarıyla tam veri kümesi üzerinde derinlemesine incelenmiştir. Bu deneyin üç konfigürasyonu karşılaştırılmıştır:

1. **UNet++ ResNet34 RGB (temel):** Yalnızca RGB girişi, tek kanallı maske çıkışı
2. **UNet++ ResNet34 RGB+SRM (çok görevli):** RGB + SRM gürültü haritası girişi, maske + kenar çok görevli çıkış
3. **UNet++ ResNet50 RGB+SRM (çok görevli):** Daha büyük encoder ile aynı çok görevli yapı

Bölme eşitliği SHA-256 hash ve satır bazlı karşılaştırmayla doğrulanmış; üç konfigürasyon özdeş eğitim-doğrulama-test bölmesini kullanmıştır.

### 5.2 Sonuçlar

**Tablo 4. Deney 3 — UNet++ konfigürasyonları karşılaştırması**

| Konfigürasyon | Giriş | Encoder | Best Epoch | Threshold | Val Dice | Test Dice | Test IoU | Test AUPRC | Görüntü F1 | Sınır F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RGB temel | RGB | ResNet34 | 38 | 0,90 | 0,5081 | 0,5259 | 0,3567 | 0,5807 | 0,7411 | 0,1995 |
| RGB+SRM çok görevli | RGB+SRM | ResNet34 | 26 | 0,90 | 0,4754 | 0,4460 | 0,2870 | 0,3448 | 0,7114 | **0,2793** |
| RGB+SRM çok görevli | RGB+SRM | ResNet50 | 14 | 0,90 | 0,4573 | 0,4525 | 0,2924 | 0,3563 | 0,7028 | 0,2703 |

Per-görüntü bazında da temel model üstündür:

| Konfigürasyon | Forged Ort. Dice | Forged Medyan Dice |
|---|---:|---:|
| RGB temel | 0,3809 | 0,4565 |
| RGB+SRM çok görevli (ResNet34) | 0,3648 | 0,4077 |
| RGB+SRM çok görevli (ResNet50) | 0,3585 | 0,3898 |

### 5.3 Yorum

En yüksek test Dice'ına (0,5259) ve AUPRC'ye (0,5807) ulaşan model yalnızca RGB girişi kullanan temel konfigürasyondur. SRM gürültü haritası ve çok görevli kenar çıkışının eklenmesi, piksel düzeyi maske örtüşmesini iyileştirmemiş; aksine hafif bir gerilemeye neden olmuştur. Bununla birlikte kenar F1 metrikleri açısından çok görevli modeller anlamlı biçimde daha iyidir (0,2793 ve 0,2703'e karşı 0,1995).

Bu bulgunun yorumu şudur: SRM gürültü haritası ve kenar kaybı, modeli manipülasyon bölgelerinin sınırlarına odaklanmaya iter. Sınır hassasiyeti belirli uygulamalarda değerli olabilir; ancak birincil metrik olan piksel düzeyi Dice açısından fazla özelleşme toplam maske örtüşmesini zayıflatmaktadır. Deney 4 ve sonrası için alınan karar, RGB-only girişi esas almak yönünde olmuştur.

---

## 6. Deney 4 – Dört Mimari Ailesi Karşılaştırması (Tam Veri Kümesi)

### 6.1 Amaç ve Tasarım

Bu deneyin amacı, farklı tasarım felsefelerini temsil eden dört mimari ailesini aynı veri bölmesi ve eğitim protokolü altında doğrudan karşılaştırmaktır. Değerlendirilen modeller:

- **UNet++ ResNet34 (CNN encoder-decoder):** İç içe geçmiş atlamalı bağlantılarla zenginleştirilmiş CNN referans modeli. ImageNet ön eğitimli ResNet34 encoder.
- **EfficientNetB0-UNet (parametre-verimli CNN):** Bileşik ölçeklendirme yöntemiyle derinlik, genişlik ve çözünürlüğü dengeli büyüten EfficientNet-B0 encoder'lı U-Net decoder.
- **SegFormer-B0 (Transformer tabanlı):** Yerel özellikler yerine uzun menzilli piksel bağımlılıklarını yakalayan hiyerarşik Transformer encoder ve basit doğrusal decoder. HuggingFace üzerinden ADE20K ön eğitimli ağırlıklar.
- **DINOv2-lite (öz-denetimli büyük model):** Büyük ölçekli öz-denetimli öğrenmeyle eğitilmiş dondurulmuş görsel omurga üzerine hafif konvolüsyonel decoder. 252×252 patch uyumlu giriş, 256×256'ya yeniden ölçeklenmiş çıkış.

Tüm modeller 256×256 değerlendirme çözünürlüğünde çalıştırılmıştır. Eğitim parametreleri Bölüm 2.4'te tanımlanan ortak protokolü izlemektedir.

### 6.2 Ana Sonuçlar

**Tablo 5. Deney 4 — Dört model test seti karşılaştırması**

| Model | Forged Dice | Forged IoU | Komp F1@0,10 | Görüntü F1 | ROC-AUC | Gerçek FP Oranı | Ort. Komp. Sayısı |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | **0,5686** | **0,3972** | **0,3577** | 0,7283 | — | **0,3051** | 1,3333 |
| DINOv2-lite | 0,5279 | 0,3586 | 0,3432 | 0,7019 | — | 0,4174 | 1,4301 |
| EfficientNetB0-UNet | 0,5216 | 0,3528 | 0,3385 | **0,7399** | — | 0,4682 | 2,2356 |
| UNet++ ResNet34 | 0,5092 | 0,3416 | 0,2956 | 0,6983 | — | 0,7182 | 3,0235 |

Ham çıktı (raw) ve küçük bileşen temizleme (clean) karşılaştırması:

| Model | Ham Dice | Temiz Dice | Δ Dice | Ham Özg. | Temiz Özg. |
|---|---:|---:|---:|---:|---:|
| SegFormer-B0 | 0,5694 | 0,5686 | −0,0007 | 0,9806 | 0,9811 |
| DINOv2-lite | 0,5271 | 0,5279 | +0,0008 | 0,9656 | 0,9670 |
| EfficientNetB0-UNet | 0,5212 | 0,5216 | +0,0004 | 0,9706 | 0,9710 |
| UNet++ ResNet34 | 0,5091 | 0,5092 | +0,0001 | 0,9577 | 0,9578 |

Görüntü düzeyi hata matrisi (ham threshold, n = 1.023):

| Model | TP | FN | TN | FP | Duyarlılık | Özgüllük |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2-lite | 544 | 7 | 17 | 455 | 0,9873 | 0,0360 |
| SegFormer-B0 | 512 | 39 | 129 | 343 | 0,9292 | 0,2733 |
| EfficientNetB0-UNet | 502 | 49 | 168 | 304 | 0,9111 | 0,3559 |
| UNet++ ResNet34 | 442 | 109 | 199 | 273 | 0,8022 | 0,4216 |

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Görüntü düzeyi hata matrislerini 2×2 ızgara şeklinde çizin. Kaynak: `deney_4/experiments_full/` içindeki model bazlı `test_metrics_summary.csv` dosyaları.

### 6.3 Maske Büyüklüğüne Göre Performans

**Tablo 6. Deney 4 — Maske büyüklük grubuna göre per-image ortalama Dice**

| Model | Q1 ≤533 px (n=138) | Q2 534–2128 px (n=138) | Q3 2129–5518 px (n=137) | Q4 >5518 px (n=138) |
|---|---:|---:|---:|---:|
| EfficientNetB0-UNet | **0,1914** | **0,2652** | 0,4194 | 0,5378 |
| UNet++ ResNet34 | 0,1784 | 0,2277 | 0,4115 | 0,5402 |
| SegFormer-B0 | 0,0812 | 0,2103 | **0,4249** | **0,5931** |
| DINOv2-lite | 0,0685 | 0,1794 | 0,4197 | 0,5431 |

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Gruplu sütun grafik: yatay eksende Q1–Q4, her grup içinde 4 model çubuğu, dikey eksen ortalama Dice. Kaynak: `analysis_review/experiment_4_forged_area_bin_analysis.csv`

### 6.4 İstatistiksel Karşılaştırma

Tüm ikili model karşılaştırmalarında hem paired t-test hem Wilcoxon testi p < 0,0001 düzeyinde anlamlıdır; bu, gözlemlenen farkların tesadüfi olmadığını göstermektedir.

### 6.5 Yorum ve Açık Sorun

Aggregate lokalizasyon (Forged Dice, Forged IoU, Bileşen F1) açısından SegFormer-B0 en güçlü modeldir. Görüntü düzeyi F1 açısından EfficientNetB0-UNet öne geçmektedir. Her iki model de anlamlı bir üstünlük kurarken ciddi bir sorun belirmiştir: gerçek görüntülerin büyük çoğunluğunda (%30–72) modeller var olmayan sahtecilik bileşenleri tespit etmiştir. DINOv2-lite bu sorunun en ağır biçimini sergilemekte; model neredeyse her görüntüyü sahte olarak nitelendirmektedir (özgüllük 0,036). Q1 küçük maskeli grubunda dört modelin tamamı anlamlı lokalizasyon sinyali üretememiştir; bu yapısal bir sınırlılık olarak bir sonraki deneye taşınmıştır.

---

## 7. Deney 5 – Kalibrasyon ve Post-processing Optimizasyonu

### 7.1 Amaç ve Tasarım

Deney 4'te tespit edilen yüksek yanlış alarm sorunu, yeniden eğitim gerektirmeden çözülebilir mi? Bu soruyu yanıtlamak için Deney 5, Deney 4'te üretilen olasılık haritalarını yeniden kullanmış ve yalnızca yorumlama katmanını sistematik biçimde optimize etmiştir. Temel güvence şudur: tüm parametre seçimleri yalnızca doğrulama seti üzerinde gerçekleştirilmiş; test setine sabit olarak uygulanmıştır.

### 7.2 Post-processing Yöntemleri

Altı farklı filtreleme yaklaşımı karşılaştırılmıştır:

1. **Ham eşikleme:** Yalnızca piksel eşiği; referans noktası
2. **Alan filtresi:** Bağlı bileşen analizi; minimum piksel alanı altındaki bileşenler silinir
3. **Ortalama olasılık filtresi:** Ortalama aktivasyonu düşük bileşenler silinir
4. **Alan + olasılık:** Her iki kriter birlikte uygulanır
5. **Morfoloji destekli filtreleme:** Açma işlemi ardından alan ve olasılık filtresi
6. **En güçlü bileşen seçimi:** Filtreleme sonrasında yalnızca belirli sayıda bileşen tutulur

### 7.3 Kalibrasyon Stratejileri

Her model için beş farklı kalibrasyon stratejisi ve bir ham referans tanımlanmıştır. Strateji seçiminin esas gerekçesi, farklı kullanım bağlamlarının farklı optimizasyon önceliklerine sahip olmasıdır. Stratejiler arasındaki dengeyi ölçmek için aşağıdaki ağırlıklı dengeli skor kullanılmıştır:

**Dengeli skor = 0,35 × Forged\_Dice + 0,25 × Komp\_F1@0,10 + 0,20 × Görüntü\_F1 + 0,20 × (1 − Gerçek\_FP\_Oranı)**

Ağırlıklandırmanın gerekçesi: Lokalizasyon (%35) çalışmanın birincil hedefidir. Bileşen tespiti (%25) her manipülasyon olayının ayrı ayrı bulunup bulunmadığını ölçer. Görüntü düzeyi karar (%20) triyaj doğruluğunu yansıtır. Gerçek görüntü güvenilirliği (%20) sistemin özgüllüğünü dengeler.

**Tablo 7. Seçilen post-processing konfigürasyonları (doğrulama seti)**

| Model | Strateji | Piksel Eşiği | Min Alan (px) | Morfoloji | Görüntü Eşiği |
|---|---|---:|---:|---|---:|
| UNet++ ResNet34 | Dengeli | 0,75 | 500 | açma k=5 + kapama k=5 | 0,000 |
| EfficientNetB0-UNet | Dengeli | 0,75 | 500 | açma k=5 | 0,705 |
| SegFormer-B0 | Dengeli | 0,75 | 500 | açma k=5 | 0,539 |
| DINOv2-lite | Dengeli | 0,75 | 500 | açma k=5 | 0,310 |
| EfficientNetB0-UNet | Düşük yanlış alarm | 0,65 | 500 | — | 0,705 |
| SegFormer-B0 | Düşük yanlış alarm | 0,75 | 500 | açma k=5 | 0,539 |

### 7.4 Test Seti Sonuçları

**Tablo 8. Deney 5 — Post-processing öncesi ve sonrası karşılaştırması**

| Model | Ham Dice | Post-proc Dice | Ham Gerçek FP | Post-proc Gerçek FP | FP Azalması |
|---|---:|---:|---:|---:|---:|
| SegFormer-B0 | 0,5583 | 0,5573 | 0,5953 | 0,2436 | −0,352 |
| EfficientNetB0-UNet | 0,5212 | 0,5199 | 0,5911 | 0,1992 | −0,392 |
| DINOv2-lite | 0,5311 | 0,5293 | 0,6504 | 0,2394 | −0,411 |
| UNet++ ResNet34 | 0,5092 | 0,5049 | 0,6780 | 0,2987 | −0,379 |

**Tablo 9. Deney 5 — Test seti dengeli strateji final sıralaması (ilk 10)**

| Sıra | Model | Strateji | Forged Dice | Komp F1@0,10 | Görüntü F1 | Gerçek FP | Final Skor |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | SegFormer-B0 | Düşük yanlış alarm | 0,5573 | 0,3263 | 0,7292 | 0,2436 | 0,5737 |
| 2 | SegFormer-B0 | Dengeli | 0,5573 | 0,3263 | 0,7292 | 0,2436 | 0,5737 |
| 3 | EfficientNetB0-UNet | Dengeli | 0,5199 | 0,3251 | 0,7400 | 0,1992 | 0,5714 |
| 4 | DINOv2-lite | Dengeli | 0,5293 | 0,3283 | 0,7032 | 0,2394 | 0,5601 |
| 5 | DINOv2-lite | Düşük yanlış alarm | 0,5296 | 0,3292 | 0,7032 | 0,2415 | 0,5600 |
| 6 | EfficientNetB0-UNet | Düşük yanlış alarm | 0,5065 | 0,3201 | 0,7400 | 0,2542 | 0,5544 |
| 7 | SegFormer-B0 | En yüksek piksel | 0,5586 | 0,3453 | 0,7292 | 0,4301 | 0,5417 |
| 8 | SegFormer-B0 | En yüksek bileşen | 0,5586 | 0,3453 | 0,7292 | 0,4301 | 0,5417 |
| 9 | EfficientNetB0-UNet | En yüksek bileşen | 0,5213 | 0,3458 | 0,7400 | 0,3856 | 0,5398 |
| 10 | UNet++ ResNet34 | Düşük yanlış alarm | 0,5049 | 0,2936 | 0,7001 | 0,2987 | 0,5304 |

**Tablo 10. Deney 5 — Görüntü düzeyi kalibrasyon metrikleri (dengeli strateji, n = 1.023)**

| Model | Brier | Kal. Hatası (ECE) | ROC-AUC | AUPRC | TP | FN | TN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNetB0-UNet | 0,290 | 0,296 | **0,799** | **0,845** | 501 | 50 | 170 | 302 |
| SegFormer-B0 | 0,291 | **0,267** | 0,761 | 0,813 | 513 | 38 | 129 | 343 |
| DINOv2-lite | 0,319 | 0,305 | 0,736 | 0,803 | 546 | 5 | 16 | 456 |
| UNet++ ResNet34 | 0,473 | 0,488 | 0,664 | 0,691 | 453 | 98 | 174 | 298 |

> **[GÖRSELLEŞTİRME ÖNERİSİ — deney5_rapor.md'den ŞEKİL 1]** Dört modelin stratejiye göre test metriklerini gösteren bar grafik serisi (Forged Dice, Bileşen F1, Görüntü F1, Gerçek FP Oranı).

> **[GÖRSELLEŞTİRME ÖNERİSİ — deney5_rapor.md'den ŞEKİL 2]** ROC ve Kesinlik-Duyarlılık eğrileri.

> **[GÖRSELLEŞTİRME ÖNERİSİ — deney5_rapor.md'den ŞEKİL 5]** Maske büyüklük grubuna göre model performansı bar grafiği.

### 7.5 Maske Büyüklüğüne Göre Performans

**Tablo 11. Deney 5 — Maske büyüklük grubuna göre per-image ortalama Dice (dengeli strateji)**

| Model | Q1 ≤533 px (n=138) | Q2 (n=138) | Q3 (n=137) | Q4 (n=138) |
|---|---:|---:|---:|---:|
| EfficientNetB0-UNet | **0,0162** | **0,1098** | 0,4106 | 0,5377 |
| DINOv2-lite | 0,0077 | 0,0983 | 0,3914 | 0,5400 |
| UNet++ ResNet34 | 0,0024 | 0,0781 | 0,3931 | 0,5259 |
| SegFormer-B0 | 0,0015 | 0,0904 | **0,4285** | **0,6010** |

### 7.6 Yorum

Post-processing'in en önemli katkısı asimetriktir: Forged Dice yalnızca 0,001–0,004 puan gerilemiş, buna karşın gerçek görüntü bileşen alarm oranı her modelde yaklaşık 35–41 puan düşmüştür. Bu asimetrinin açıklaması mantıksal olarak tutarlıdır: filtrelenen küçük bileşenler zaten gerçek sahtekârlık maskeleriyle örtüşmeyen bölgelerde bulunmaktaydı; dolayısıyla onları silmek lokalizasyon kaybına neden olmadı.

Bu deneyin çözümsüz bıraktığı sorun, Q1 küçük maskeli grupta Dice değerlerinin 0,002–0,016 aralığında kalmaya devam etmesidir. Post-processing katmanı bu yapısal zayıflığı gideremez; çünkü sorun modelin olasılık haritasında küçük manipülasyon bölgeleri için anlamlı sinyal üretememesinden kaynaklanmaktadır. Bu sinyal, 256×256 çözünürlükte o kadar küçük bir bölge için kaçınılmaz biçimde gürültüde kaybolmaktadır.

DINOv2-lite'ın görüntü düzeyi kalibrasyon sorununun post-processing katmanıyla çözülemediği gösterilmiştir (TN = 16, özgüllük ≈ 0,034). Bu sonuç, sorunun dondurulmuş omurganın özellik üretim biçiminden kaynaklandığına işaret etmektedir ve Deney 6'da DINOv2-lite'ın dışarıda bırakılmasının gerekçesini oluşturmuştur.

---

## 8. Deney 6 – 384×384 Çözünürlük ve Küçük Maske Lokalizasyonu

### 8.1 Araştırma Sorusu ve Tasarım

Deney 5'te açık kalan temel soru şudur: 256×256 değerlendirme çözünürlüğünde yapısal sinyal kaybı yaşayan küçük manipülasyon bölgeleri için ne yapılabilir? 384×384 çalışma çözünürlüğünde eğitim, aynı gerçek boyuttaki sahtecilik bölgesinin olasılık haritasında daha fazla piksel ile temsil edilmesini sağlar. Dolayısıyla bu deney şu soruyu yanıtlamaya çalışmaktadır: **giriş ve tahmin çözünürlüğünü 384×384'e yükseltmek, küçük manipülasyon bölgelerinin olasılık haritasında daha güçlü ve daha doğru konumlanmış sinyal üretmesini sağlar mı?**

Bu soruyu yanıtlamak için Deney 4 ve 5'te güçlü aday olarak belirlenen iki model — EfficientNetB0-UNet ve SegFormer-B0 — yeni çözünürlükte sıfırdan yeniden eğitilmiştir. DINOv2-lite'ın görüntü düzeyi kalibrasyon sorununun omurgadan kaynaklandığı Deney 5'te gösterildiğinden bu deneye dahil edilmemiştir.

### 8.2 Eğitim Ayrıntıları

Eğitim izleme skoru olarak validation üzerinde `0,50 × forged_dice + 0,30 × Q1_dice + 0,20 × Q2_dice` bileşik metriği kullanılmıştır. Bu ağırlıklandırma, modeli küçük maske performansını doğrudan optimize etmeye yönlendirir. Pozitif piksel dengesizliği `pos_weight` ile ele alınmış; oran yaklaşık %2 civarında olduğu için pos_weight üst sınır olan 20 değerine ulaşmıştır. Post-processing parametreleri ve eşikler yalnızca doğrulama seti üzerinde seçilmiştir; test setinde hiçbir ayar yapılmamıştır.

**Tablo 12. Deney 6 — Model ve eğitim özeti**

| Model | Tip | Toplam Parametre | Epoch | Best Epoch | Val Forged Dice | Val Q1 Dice | Val Q2 Dice |
|---|---|---:|---:|---:|---:|---:|---:|
| EfficientNetB0-UNet 384 | CNN encoder-decoder | 6.251.469 | 32 | 24 | 0,396 | 0,320 | 0,328 |
| SegFormer-B0 384 | Transformer | 3.714.401 | 40 | 39 | 0,430 | 0,319 | 0,335 |

Validation sürecinde Q1 Dice 0,30 bandına ulaşması, eğitim sırasında küçük maske sinyalinin oluştuğunu doğrulamaktadır.

### 8.3 Test Seti Sonuçları

**Tablo 13. Deney 6 — Strateji bazlı test seti sonuçları**

| Model | Strateji | Final Skor | Forged Dice | Q1 Dice | Q2 Dice | Komp F1@0,10 | Gerçek FP | Görüntü F1 | Dice<0,05 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 384 | balanced\_final\_score | 0,486 | **0,645** | 0,277 | 0,325 | **0,438** | 0,356 | 0,762 | 163 |
| SegFormer-B0 384 | small\_object\_practical | 0,484 | 0,645 | 0,285 | 0,353 | 0,418 | 0,400 | 0,762 | 149 |
| EfficientNetB0-UNet 384 | balanced\_final\_score | 0,481 | 0,577 | 0,283 | 0,317 | 0,421 | **0,239** | **0,788** | 165 |
| EfficientNetB0-UNet 384 | low\_false\_alarm | 0,481 | 0,577 | 0,283 | 0,317 | 0,421 | **0,239** | **0,788** | 165 |
| EfficientNetB0-UNet 384 | best\_small\_mask\_q1 | 0,440 | 0,557 | **0,300** | 0,362 | 0,268 | 0,472 | 0,788 | 130 |
| SegFormer-B0 384 | best\_small\_mask\_q1 | 0,431 | 0,560 | 0,307 | 0,353 | 0,324 | 0,631 | 0,762 | 102 |

### 8.4 Deney 5 ile Karşılaştırma — Balanced Strateji Bazında

**Tablo 14. Deney 6 → Deney 5 (balanced strateji) Δ değerleri**

| Model | Δ Forged Dice | Δ Q1 Dice | Δ Q2 Dice | Δ Komp F1 | Δ Gerçek FP | Δ Görüntü F1 | Δ Dice<0,05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientNetB0-UNet | +0,057 | **+0,267** | +0,208 | +0,096 | +0,040 | +0,048 | −113 |
| SegFormer-B0 | +0,088 | **+0,275** | +0,235 | +0,112 | +0,112 | +0,033 | −125 |

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Deney 5 vs Deney 6 Q1–Q4 Dice karşılaştırmasını gruplu çubuk grafikle gösterin. Kaynak: `analysis_review/experiment_6_vs_experiment_5_reconstructed_comparison.csv`

### 8.5 Küçük Maske Trade-off Analizi

Küçük maske yakalama oranını maksimize eden stratejiler (`best_small_mask_q1`) daha düşük piksel eşiği ve daha az bileşen temizliği kullandığından gerçek görüntü alarm oranını belirgin biçimde artırmaktadır. SegFormer'ın Q1 odaklı stratejisinde gerçek FP oranı 0,631'e, EfficientNet'inki 0,472'ye yükselmiştir. Dengeli stratejilerde ise Q1 Dice yaklaşık 0,28 bandında kalırken gerçek FP oranı kabul edilebilir düzeyde tutulabilmektedir. Bu trade-off, nihai model kararını doğrudan etkileyen temel gerilimdir.

### 8.6 Yorum

Çözünürlük artışının küçük maske performansına etkisi istatistiksel olarak son derece güçlüdür (bkz. Bölüm 9.6). Q1 Dice değerleri EfficientNetB0-UNet için 0,016'dan 0,283'e, SegFormer-B0 için 0,002'den 0,277'ye yükselmiştir. Bu iyileşmenin yalnızca post-processing seçiminin yan etkisi olmadığı, "best_forged_dice" strateji karşılaştırmalarında da aynı yönde büyük farklılıklar görüldüğü doğrulanmıştır.

Deney 6'nın iki final aday modeli belirlemesi, iki farklı operasyonel önceliği temsil eden rollere karşılık gelir: SegFormer-B0 384 lokalizasyon kalitesini; EfficientNetB0-UNet 384 ise false alarm maliyetini optimize eder.

---

## 9. Final Analiz – İki Final Model Kapsamlı Değerlendirmesi

### 9.1 Final Analizin Amacı ve Protokolü

Final analiz, Deney 6 sonrasında seçilen iki aday modeli yeni bir eğitim yapmadan bağımsız bir test seti üzerinde kapsamlı biçimde değerlendirir. Değerlendirilen boyutlar: temiz PNG test seti, dağılım kayması altında dayanıklılık (robustness), küçük maske performansı, bileşen yakalama, görüntü düzeyi karar ve failure case analizi.

**Temel protokol güvencesi:** Eşik ve post-processing parametreleri test setinde seçilmemiştir. Parametreler önceki doğrulama seçimlerinden okunmuş, test ve robustness koşullarında sabit tutulmuştur.

### 9.2 Final Aday Modeller

**Tablo 15. Final aday modeller ve seçilen stratejiler**

| Model | Rol | Boyut | Strateji | Parametre Sayısı |
|---|---|---:|---|---:|
| SegFormer-B0 384 | Lokalizasyon odaklı final model | 384×384 | balanced\_final\_score | 3.714.401 |
| EfficientNetB0-UNet 384 | Muhafazakâr / düşük alarm final model | 384×384 | low\_false\_alarm | 6.251.469 |

### 9.3 Post-processing Pipeline

**Tablo 16. Her iki final model için sabit post-processing parametreleri**

| Parametre | SegFormer-B0 384 | EfficientNetB0-UNet 384 |
|---|---|---|
| Piksel eşiği | 0,85 | 0,85 |
| Görüntü eşiği | 0,59 | 0,78 |
| Post-processing modu | morph\_area\_probability\_clean | morph\_area\_probability\_clean |
| Min. bileşen alanı (px) | 100 | 100 |
| Morfoloji | open\_close | open\_close |
| Kernel boyutu | 5×5 | 5×5 |
| Görüntü skoru | max\_probability | max\_probability |

Ölçüm akışı şöyledir: (1) Olasılık haritası piksel bazında `prob ≥ piksel_eşiği` ile ikili maskeye çevrilir. (2) `open_close` morfoloji işlemi 5×5 kernel ile uygulanır. (3) Bağlı bileşenler 8-komşulukla çıkarılır. (4) Alanı 100 pikselden küçük bileşenler elenir. (5) Görüntü düzeyi skor `max_probability` olarak hesaplanır. (6) Görüntü kararı `görüntü_skoru ≥ görüntü_eşiği` ile verilir.

### 9.4 Clean Test Sonuçları

**Tablo 17. Final model karşılaştırması — clean test seti + 256×256 referanslar**

| Model | Boyut | Strateji | Forged Dice | Forged IoU | Ort. Forged Dice | Medyan Forged Dice | Komp F1@0,10 | Gerçek FP | Görüntü F1 | ROC-AUC | AUPRC | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SegFormer-B0 384** | 384 | balanced | **0,6451** | **0,4761** | 0,4334 | 0,5257 | **0,4384** | 0,3559 | 0,7620 | 0,8503 | 0,8857 | 0,2639 | 0,2659 |
| **EfficientNetB0-UNet 384** | 384 | low\_fa | 0,5770 | 0,4055 | 0,3954 | 0,4797 | 0,4206 | **0,2394** | **0,7878** | **0,8616** | **0,8890** | **0,2417** | **0,2578** |
| SegFormer-B0 256 | 256 | balanced | 0,5573 | 0,3863 | — | — | 0,3263 | 0,2436 | 0,7292 | 0,7613 | 0,8128 | 0,2906 | 0,2673 |
| DINOv2-lite 256 | 256 | balanced | 0,5293 | 0,3599 | — | — | 0,3283 | 0,2394 | 0,7032 | 0,7364 | 0,8031 | 0,3188 | 0,3052 |
| EfficientNetB0-UNet 256 | 256 | balanced | 0,5199 | 0,3512 | — | — | 0,3251 | 0,1992 | 0,7400 | 0,7995 | 0,8446 | 0,2898 | 0,2955 |
| UNet++ ResNet34 256 | 256 | ham ref. | 0,5092 | 0,3415 | — | — | 0,2246 | 0,6780 | 0,6985 | 0,6991 | 0,7377 | 0,3031 | 0,2742 |

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Altı modelin Forged Dice, Bileşen F1, Görüntü F1 ve Gerçek FP oranını gösteren gruplu çubuk grafik. Kaynak: `final_analysis/experiments_full/final_analysis/final_model_comparison.csv`

> **[GÖRSELLEŞTİRME ÖNERİSİ]** ROC ve Kesinlik-Duyarlılık eğrileri (iki final model). Kaynak: `final_analysis/experiments_full/final_analysis/plots/roc_pr_curves_final.png`

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Güven kalibrasyonu (reliability diagram). Kaynak: `final_analysis/experiments_full/final_analysis/plots/reliability_diagram_final.png`

### 9.5 Küçük Maske (Q1–Q4) Analizi

**Tablo 18. Final modeller — maske büyüklük grubuna göre lokalizasyon performansı**

| Model | Quartile | n | Ort. Dice | Medyan Dice | Ort. IoU | Medyan IoU | Dice < 0,05 |
|---|---|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 384 | Q1 | 138 | 0,2767 | 0,2053 | 0,1957 | 0,1144 | 65 |
| SegFormer-B0 384 | Q2 | 138 | 0,3249 | 0,3358 | 0,2386 | 0,2018 | 53 |
| SegFormer-B0 384 | Q3 | 137 | 0,4399 | 0,5266 | 0,3291 | 0,3574 | 35 |
| SegFormer-B0 384 | Q4 | 138 | 0,6922 | 0,7363 | 0,5770 | 0,5827 | 10 |
| EfficientNetB0-UNet 384 | Q1 | 138 | 0,2833 | 0,2320 | 0,2003 | 0,1313 | 64 |
| EfficientNetB0-UNet 384 | Q2 | 138 | 0,3174 | 0,3204 | 0,2303 | 0,1909 | 57 |
| EfficientNetB0-UNet 384 | Q3 | 137 | 0,4127 | 0,4821 | 0,2990 | 0,3176 | 33 |
| EfficientNetB0-UNet 384 | Q4 | 138 | 0,5685 | 0,5737 | 0,4320 | 0,4022 | 11 |

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Q1–Q4 grubuna göre iki final modelin ortalama Dice değerlerini gösteren yan yana çubuk grafik. Kaynak: `final_analysis/experiments_full/final_analysis/plots/small_mask_quartile_final.png`

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Per-image Dice dağılımını gösteren violin veya kutu grafiği. Kaynak: `final_analysis/experiments_full/final_analysis/plots/per_image_dice_distribution.png`

Q1 grubunda ortalama Dice değerleri her iki model için de yaklaşık 0,28 düzeyindedir ve Q1 vakalarının yaklaşık yarısı (65 ve 64 vaka) Dice < 0,05 eşiğinin altında kalmaktadır. Bu, 384×384 çözünürlük artışının Q1 sorununu belirgin biçimde hafiflettiğini ancak tamamen çözmediğini göstermektedir. Maske alanı büyüdükçe başarı belirgin biçimde artmakta; SegFormer-B0 384 Q4'te 0,6922 ortalama ve 0,7363 medyan Dice üretmektedir.

### 9.6 Robustness Analizi

Robustness analizi, clean PNG testine ek olarak 8 farklı bozulma koşulunda gerçekleştirilmiştir. Görseller bellekte bozulmuş, maskeler değiştirilmemiş ve threshold ayarları yeniden seçilmemiştir.

Bozulma koşulları: JPEG Q90, JPEG Q70, JPEG Q50, Gaussian blur light (3×3, sigma 0,8), Gaussian blur medium (5×5, sigma 1,2), Gaussian noise light (std 0,02), Gaussian noise medium (std 0,05), JPEG70 + blur light (birleşik).

**Tablo 19. SegFormer-B0 384 robustness sonuçları**

| Koşul | Forged Dice | Q1 Dice | Komp F1@0,10 | Gerçek FP | Görüntü F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Clean PNG | 0,6451 | 0,2767 | 0,4384 | 0,3559 | 0,7620 | 0,8503 |
| JPEG Q90 | 0,6438 | 0,2733 | 0,4349 | 0,3602 | 0,7543 | 0,8443 |
| JPEG Q70 | 0,6424 | 0,2496 | 0,4226 | 0,4301 | 0,7401 | 0,8208 |
| JPEG Q50 | 0,6346 | 0,2203 | 0,4078 | 0,4470 | 0,7391 | 0,8043 |
| Blur light | 0,6505 | 0,2718 | 0,4398 | 0,3496 | 0,7651 | 0,8476 |
| Blur medium | 0,6502 | 0,2606 | 0,4361 | 0,3178 | 0,7638 | 0,8420 |
| Noise light | 0,6367 | 0,2567 | 0,4177 | 0,4280 | 0,7390 | 0,8179 |
| Noise medium | 0,6175 | 0,1767 | 0,3629 | 0,5148 | 0,7294 | 0,7627 |
| JPEG70 + Blur light | 0,6453 | 0,2459 | 0,4242 | 0,4068 | 0,7411 | 0,8227 |

**Tablo 20. EfficientNetB0-UNet 384 robustness sonuçları**

| Koşul | Forged Dice | Q1 Dice | Komp F1@0,10 | Gerçek FP | Görüntü F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Clean PNG | 0,5770 | 0,2833 | 0,4206 | 0,2394 | 0,7878 | 0,8616 |
| JPEG Q90 | 0,5715 | 0,2690 | 0,4104 | 0,2585 | 0,7793 | 0,8517 |
| JPEG Q70 | 0,5726 | 0,1732 | 0,3923 | 0,2521 | 0,7624 | 0,8302 |
| JPEG Q50 | 0,5675 | 0,1042 | 0,3640 | 0,2521 | 0,7512 | 0,8057 |
| Blur light | 0,5796 | 0,2649 | 0,4186 | 0,2246 | 0,7854 | 0,8595 |
| Blur medium | 0,5734 | 0,2348 | 0,4115 | 0,2076 | 0,7752 | 0,8512 |
| Noise light | 0,5569 | 0,2186 | 0,3852 | 0,3220 | 0,7643 | 0,8132 |
| Noise medium | 0,5442 | 0,1371 | 0,3397 | 0,3686 | 0,7335 | 0,7548 |
| JPEG70 + Blur light | 0,5749 | 0,1665 | 0,3930 | 0,2267 | 0,7732 | 0,8360 |

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Bozulma koşullarına göre Forged Dice, Q1 Dice, Gerçek FP oranı ve Bileşen F1 çizgi grafikleri (her metrik için ayrı panel, iki model aynı grafikte). Kaynak: `final_analysis/experiments_full/final_analysis/plots/` içindeki `robustness_*.png` dosyaları.

**Robustness yorumu:** JPEG sıkıştırma ve gürültü bozulmaları Q1 küçük maskelerde en sert düşüşü üretmektedir. SegFormer-B0 384 aggregate Forged Dice açısından bozulmalara daha dayanıklı görünmektedir; ancak gürültü medium koşulunda gerçek FP oranı 0,5148'e yükselir. EfficientNetB0-UNet 384, temiz ve pek çok bozulma koşulunda daha düşük gerçek FP oranını korur; buna karşın Q1 Dice, JPEG Q70/Q50 ve birleşik koşullarda daha sert düşer. Hafif bulanıklık (blur light) her iki modelin Forged Dice'ını clean koşuluna göre hafif artırmakta; bu, hafif düzleşmenin keskin sınır piksellerindeki gürültüyü azalttığını ve bazı vakalarda maskenin daha düzgün segmentasyonuna yardımcı olduğunu düşündürmektedir.

### 9.7 İstatistiksel Karşılaştırma

**Tablo 21. Final iki model arası istatistiksel testler (forged görüntüler, n = 551)**

| Karşılaştırma | Metrik | Alt Küme | n | Ort. Fark | %95 Bootstrap GA | Paired-t p | Wilcoxon p | Cohen's d |
|---|---|---|---:|---:|---|---:|---:|---:|
| SegFormer 384 − EfficientNet 384 | Dice | Forged (tümü) | 551 | +0,0380 | [0,0220; 0,0548] | 6,30×10⁻⁶ | 8,28×10⁻⁵ | 0,194 |
| SegFormer 384 − EfficientNet 384 | IoU | Forged (tümü) | 551 | +0,0448 | [0,0302; 0,0596] | 7,35×10⁻⁹ | 6,63×10⁻⁶ | 0,250 |
| SegFormer 384 − EfficientNet 384 | Dice | Q1 | 138 | −0,0066 | [−0,0354; 0,0209] | 0,643 | 0,161 | −0,039 |
| SegFormer 384 − EfficientNet 384 | Dice | Q2 | 138 | +0,0076 | [−0,0218; 0,0364] | 0,613 | 0,529 | 0,043 |

**Tablo 22. 384 vs 256 çözünürlük etkisi — istatistiksel testler**

| Karşılaştırma | Metrik | Alt Küme | Ort. Fark | %95 Bootstrap GA | Paired-t p | Cohen's d |
|---|---|---|---:|---|---:|---:|
| SegFormer 384 − SegFormer 256 | Dice | Forged | +0,1533 | [0,1342; 0,1734] | 5,97×10⁻⁴³ | 0,640 |
| SegFormer 384 − SegFormer 256 | Dice | Q1 | +0,2444 | [0,1948; 0,2935] | 3,59×10⁻¹⁷ | 0,823 |
| EfficientNet 384 − EfficientNet 256 | Dice | Forged | +0,1271 | [0,1066; 0,1474] | 2,17×10⁻²⁹ | 0,509 |
| EfficientNet 384 − EfficientNet 256 | Dice | Q1 | +0,2147 | [0,1652; 0,2637] | 6,37×10⁻¹⁴ | 0,712 |

Bootstrap güven aralıkları 5.000 iterasyonla hesaplanmıştır.

**İstatistiksel yorum:** SegFormer-B0 384 ile EfficientNetB0-UNet 384 arasında genel forged lokalizasyon için SegFormer lehine istatistiksel olarak anlamlı fark mevcuttur (p < 10⁻⁵). Ancak en küçük maske çeyreğinde (Q1) iki model arasındaki fark anlamlı değildir (p = 0,643). 384×384 final eğitimlerinin kendi 256×256 referanslarına göre forged ve Q1 Dice metriklerinde istatistiksel olarak son derece güçlü artış sağladığı gösterilmiştir.

### 9.8 Failure Case Analizi

Her final model için 7 failure/success grubu oluşturulmuştur (her grupta en fazla 12 örnek):

| Grup | Anlam |
|---|---|
| best\_forged\_cases | Forged görüntülerde en yüksek Dice örnekleri |
| worst\_forged\_cases | Forged görüntülerde en düşük Dice örnekleri |
| small\_mask\_failures | Q1/Q2 maskeli, Dice < 0,05 küçük maske hataları |
| false\_positive\_authentic | Gerçek görüntülerde tahmin bileşeni üretilen örnekler |
| false\_negative\_forged | Forged olduğu halde görüntü düzeyinde kaçırılan örnekler |
| large\_mask\_success | Q4 büyük maskeli başarılı örnekler |
| large\_mask\_failure | Q4 büyük maskeli başarısız örnekler |

Model anlaşmazlık analizi: SegFormer'ın EfficientNet'e göre çok daha iyi performans gösterdiği 12 ve EfficientNet'in SegFormer'a göre çok daha iyi olduğu 12 vaka ayrı olarak incelenmiştir. En büyük SegFormer lehine Dice farkı +0,8976; en büyük EfficientNet lehine fark −0,7144'tür.

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Failure case örneklerini orijinal görüntü + ground truth maske + model tahmin maskesi üçlüsü şeklinde grid görsel olarak ekleyin. Kaynak: `final_analysis/experiments_full/final_analysis/segformer_b0_rgb_384_smallmask/failure_cases/` ve `efficientnetb0_unet_rgb_384_smallmask/failure_cases/` klasörleri.

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Model anlaşmazlık vakası örnekleri. Kaynak: `final_analysis/experiments_full/final_analysis/plots/model_disagreement_examples.png`

> **[GÖRSELLEŞTİRME ÖNERİSİ]** Hata matrisleri (iki final model). Kaynak: `final_analysis/experiments_full/final_analysis/plots/confusion_matrices_final.png`

---

## 10. Sonuç ve Tartışma

### 10.1 Altı Deneyin Ana Bulguları

Çalışma boyunca izlenen deneysel yol birkaç temel öğrenme noktasını ortaya koymuştur.

**Mimari seçiminin önemi.** 15 model arasında yalnızca üçü anlamlı lokalizasyon başarısı sergilemiştir. Kopyala-yapıştır manipülasyon tespitine özgü modeller ve geleneksel adli araçlar bu görevde yetersiz kalmıştır; bu, bilimsel görüntü sahtekârlığının farklı görsel izler bıraktığını ve genel amaçlı segmentasyon mimarilerinin daha uygun bir temel sağladığını göstermektedir.

**RGB-only girişin yeterliliği.** Deney 3, SRM gürültü haritası ve kenar kaybı eklemenin piksel düzeyi lokalizasyon açısından net bir kazanım sağlamadığını ortaya koymuştur. Çok görevli modeller sınır hassasiyetinde ilerleme kaydederken ana metrik olan Dice'ta gerileme yaşanmıştır.

**Post-processing'in asimetrik katkısı.** Deney 5 post-processing optimizasyonunun Dice kaybı olmaksızın gerçek görüntü alarm oranını 35–41 puan düşürebildiğini göstermiştir. Bu asimetri, filtrelenen küçük bileşenlerin gerçek sahtekârlık maskeleriyle zaten örtüşmediğini ve dolayısıyla silinmelerinin lokalizasyon kaybına yol açmadığını kanıtlamaktadır.

**Çözünürlük artışının küçük maske etkisi.** Deney 6'nın en önemli bulgusu, 384×384 çözünürlükte eğitimin Q1 küçük maske Dice değerlerini yaklaşık 10–15 kat artırdığıdır. Bu iyileşme istatistiksel olarak son derece güçlüdür (p < 10⁻¹³) ve post-processing seçiminin değil, olasılık haritasındaki sinyal kalitesinin bir yansımasıdır.

**İki modelin tamamlayıcı rolleri.** Final analiz, iki modelin farklı operasyonel öncelikleri temsil ettiğini net biçimde göstermiştir. SegFormer-B0 384 aggregate lokalizasyon metriklerinde üstündür; EfficientNetB0-UNet 384 ise false alarm kontrolü ve görüntü düzeyi kalibrasyon açısından daha güvenilir bir profil sergiler.

### 10.2 Lokalizasyon Başarısı ve False Alarm Gerilimi

Birincil metrik piksel/bileşen lokalizasyonu ise final model olarak **SegFormer-B0 384** önerilmelidir: Forged Dice 0,6451, Forged IoU 0,4761, Bileşen F1@0,10 0,4384.

Pratik uygulamada düşük yanlış alarm ve görüntü düzeyi güvenilirlik öncelikliyse **EfficientNetB0-UNet 384** daha muhafazakâr bir seçimdir: Gerçek FP oranı 0,2394, Görüntü F1 0,7878, ROC-AUC 0,8616, AUPRC 0,8890.

Bu iki model tezde birlikte raporlanmalıdır; çünkü biri lokalizasyon kalitesini, diğeri false alarm maliyetini optimize eden iki farklı operasyonel önceliği temsil etmektedir.

### 10.3 Küçük Maske Sorununun Kısmi Çözümü

Q1 grubundaki Dice değerleri Deney 4'teki ~0,002–0,016 bandından Deney 6/Final Analiz'deki ~0,28 bandına yükselmiştir. Bu önemli bir ilerleme olmakla birlikte Q1 vakalarının yaklaşık yarısı hâlâ Dice < 0,05 eşiğinin altında kalmaktadır. Küçük maske sorunu çözülmüş değil, kısmen hafifletilmiştir. 384×384 çözünürlük, problemi yapısal olarak ortadan kaldırmamakta; yalnızca sinyal-gürültü oranını iyileştirmektedir.

---

## 11. Sınırlılıklar ve Gelecek Çalışma Önerileri

### 11.1 Sınırlılıklar

**Robustness checkpoint bağımlılığı.** Robustness analizi, model checkpoint'inin mevcut olmasını gerektirmektedir. Checkpoint eksikliğinde clean analiz önbelleklenmiş olasılık haritasıyla tamamlanabilmekte ancak bozuk görüntü üzerinde ileri geçiş yapılamamaktadır.

**McNemar testi güvenilirlik notu.** `mcnemar_tests.csv` dosyasında görüntü düzeyi eşleşme `n = 1.967` olarak görünmektedir; test seti 1.023 görüntüdür. Birleştirme anahtarı olarak `image_id` kullanıldığında, aynı `image_id` değerinin gerçek ve sahte klasörlerde birlikte bulunabilmesi satır çoğalmasına neden olabilir. Bu nedenle tez raporunda McNemar sonucu kullanılmadan önce `sample_id` veya `case_key` ile yeniden doğrulama önerilir.

**Tek veri kümesi.** Sonuçlar yalnızca ReCodAI-LUC veri kümesi üzerinde geçerlidir. Farklı bilimsel görüntü türlerine veya manipülasyon yöntemlerine genellenebilirlik için bağımsız veri kümeleri üzerinde doğrulama gerekmektedir.

**Threshold sabitliği.** Tüm threshold ve post-processing parametreleri doğrulama seti üzerinde seçilmiş ve test setinde sabit tutulmuştur. Test seti dağılımının doğrulama setinden belirgin biçimde farklılaşması durumunda sabit threshold'lar optimal olmayabilir.

### 11.2 Gelecek Çalışma Önerileri

**DINOv2-lite sınırlı ince ayar.** DINOv2-lite'ın görüntü düzeyi kalibrasyon sorunu, dondurulmuş omurga özellik üretiminden kaynaklanmaktadır. Omurganın son birkaç katmanının serbest bırakılarak ince ayar yapılması bu sorunu giderebilir ve foundation model özelliklerinin bu göreve aktarılmasını iyileştirebilir.

**Daha güçlü domain augmentasyon.** Eğitim sırasında JPEG sıkıştırma, bulanıklık ve gürültü augmentasyonlarının sistematik olarak eklenmesi, robustness analizinde gözlemlenen bozulma duyarlılığını azaltabilir.

**Validation-time threshold adaptasyonu.** Bozulma koşuluna göre threshold'u otomatik olarak uyarlayan mekanizmalar, robustness performansını iyileştirebilir.

**Uncertainty-aware post-processing.** Model tahmin güveni üzerine kurulu belirsizlik tahmini, özellikle Q1 küçük maske vakalarında daha bilinçli filtreleme kararları almayı sağlayabilir.

**Daha büyük encoder veya yüksek çözünürlük.** 512×512 veya daha büyük encoder varyantlarının (SegFormer-B2, SegFormer-B4) Q1 performansını daha da artırıp artırmayacağı araştırılmaya değerdir; bu büyüme hesaplama maliyetini artıracak olsa da küçük maske sorununa yönelik sistematik bir çözüm adayıdır.

---

*Bu rapor, `recod_luc_final_analysis.py` ve `recod_luc_final_analysis.ipynb` çıktılarıyla `analysis_review/` klasöründeki deney analizleri temel alınarak hazırlanmıştır. Sayısal değerler ilgili CSV dosyalarından doğrudan alınmıştır.*
