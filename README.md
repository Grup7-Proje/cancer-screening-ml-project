<div align="center">
  <img width="480" height="320" alt="Flag_of_Turkey svg" src="https://github.com/user-attachments/assets/3eabddf6-0791-4fdf-a1a2-1241124a0aa3" />
</div>  
<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=500&size=26&pause=1000&color=007ACC&center=true&vCenter=true&width=800&lines=Cancer+Screening+ML+Project;Erken+Teşhis+İçin+Makine+Öğrenmesi;KOÜ+%7C+MCBÜ+%7C+HİTÜ" alt="Proje Başlığı Animasyonu" />
  
  <p><b>Kocaeli Üniversitesi, Manisa Celal Bayar Üniversitesi ve Hitit Üniversitesi öğrencileri işbirliği ile geliştirilmiştir.</b></p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Machine_Learning-FF9900?style=for-the-badge" alt="Machine Learning" />
  <img src="https://img.shields.io/badge/CDC_BRFSS_Data-007ACC?style=for-the-badge" alt="Data" />
  <img src="https://img.shields.io/badge/Academic_Research-4CAF50?style=for-the-badge" alt="Academic Research" />
</p>

---

## Proje Hakkında

Bu proje, kanser taraması, risk faktörlerinin tespiti ve erken teşhis süreçlerini optimize etmek amacıyla geliştirilmiş bir makine öğrenmesi araştırmasıdır. Hastaların demografik bilgileri, yaşam tarzı alışkanlıkları ve genel sağlık durumlarına ilişkin veriler analiz edilerek, bireylerin kanser riski taşıyıp taşımadığı veya kanser tarama programlarına uyum gösterip göstermeyeceği tahmin edilmektedir.

Üç farklı üniversitenin bilgi birikimi ve işbirliği ile disiplinlerarası bir akademik çalışma olarak hayata geçirilen bu proje, koruyucu hekimlik politikaları için veri odaklı bir karar destek mekanizması sunmayı hedeflemektedir.

## Veri Seti: 2024 BRFSS (SAS Transport Format)

Çalışmamızda **CDC (Centers for Disease Control and Prevention)** tarafından yayınlanan [2024 BRFSS (Behavioral Risk Factor Surveillance System)](https://www.cdc.gov/brfss/index.html) veri seti kullanılmıştır. 

* **İçerik:** Kanser öyküsü (cilt kanseri ve diğer kanser türleri), sigara/alkol tüketimi, fiziksel aktivite, kronik hastalık geçmişi ve sosyodemografik anket cevapları.
* **Veri Boyutu:** Veri seti oldukça büyük hacimli olduğundan (orijinal .xpt ve işlenmiş .csv dosyaları), GitHub reposunda barındırılmamaktadır (Bkz. *data/drive.txt*).

## Metodoloji

Tabular anket verilerinden anlamlı sonuçlar üretebilmek için aşağıdaki veri bilimi boru hattı (pipeline) izlenmiştir:

1. **Veri Okuma ve Dönüştürme:** CDC BRFSS 2024 veri setinin Codebook'u, veri mimarlarınca tek tek incelenmiş olup, ana ve bağımsız değişkenler kararlaştırılmıştır.
<div align="center">
  <img width="921" height="513" alt="CHCOCNC1" src="https://github.com/user-attachments/assets/58d2455a-a160-45a1-b2bb-d901017832e6" />
  <p>resim 1.0 "Ana Değişken CHCOCNC1"</p>
  <img width="885" height="489" alt="SMOKE100" src="https://github.com/user-attachments/assets/5b574605-cd30-45c4-aadc-d14a90f1b7a6" />
  <p>resim 2.0 "Yan Değişken Örn. SMOKE100"</p>
</div> 

3. SAS `.xpt` formatındaki verilerin Python ortamına aktarılması ve Pandas DataFrame formatına dönüştürülmesi.
4. **Veri Ön İşleme (Data Preprocessing):** 
   * Ana ve bağımsız değişkenlerin belirlenmesi.
   ```
   Ana Değişkenler:
   CHCOCNC1: Melanom veya herhangi bir kanser teşhisi konuldu mu? (30 kategori)

   Bağımsız Değişkenler:
   CHCSCNC1: Cilt kanseri veya melanom teşhisi konuldu mu?
   CHECKUP1: Rutin kontrolünüz için en son ne zaman doktora gittiniz?
   _AGEG5YR: Kaç yaşınızdasınız? (14 kategori)
   SMOKE100: Hayatınız boyunca en az 100 sigara içtniz mi?
   DIABETE4: Daha önce diyabet teşhisi konuldu mu?
   EXERANY2: Geçtiğimiz ay boyunca, işiniz hariç, fiziksel aktivite veya egzersiz yaptınız mı?
   ASTHMA3:  Daha önce astım teşhisi konuldu mu?
   CHCKDNY2: Size hiç Böbrek taşı, mesane enfeksiyonu veya idrar kaçırmak hariç böbrek hastalığı teşhisi konuldu mu?
   HAVARTH4: Size hiç artrit, romatoid artrit, gut, lupus veya fibromiyalji gibi bir rahatsızlığınız olduğu söylendi mi?
   CVDINFR4: Daha önce kalp krizi (miyokard enfarktüsü) geçirdiniz mi?
   PERSDOC3: Bir veya bir grup doktorun kişisel sağlık sağlayıcınız olduğunu düşünüyor musunuz?
   _RFHLTH:  Sağlık durum sorusu
   INCOME3:  Tüm gelir kaynaklarınızdan gelen yllık geliriniz ne kadar? (11 kategori)
   EDUCA:    Tamamladığınız en yüksek eğitim durumunuz nedir?
   _BMI5CAT: Vücut Kitle Endeksiniz nedir? (4 categories of BMI)
   ```
   * BRFSS özelindeki "Bilmiyorum/Reddedildi" (örn: 77, 99 kodlu) yanıtlarının eksik veri (NaN) olarak ele alınması.
   * Hedef değişkenin (kanser tanısı / tarama durumu) belirlenmesi ve sınıf dengesizliklerinin (SMOTE vb. yöntemlerle) giderilmesi.
5. **Özellik Mühendisliği (Feature Engineering):** Genel kanser teşhisi ile en çok korelasyon gösteren demografik ve davranışsal özelliklerin (Örn: `_AGEG5YR`, `SMOKE100`, `CHCSCNCR`) seçilmesi.
6. **Modelleme:** Tabular verilerde yüksek performans gösteren Lojistik Regresyon, Bayesyen Yaklaşım ve Yapay Sinir Ağları kullanılarak oluşturulan modellerin eğitilmesi:
7. **Değerlendirme (Evaluation):** Modeller; Doğruluk (Accuracy), Hassasiyet (Precision), Duyarlılık (Recall), F1-Skoru ve ROC-AUC metrikleri ile istatistiksel olarak ölçülmüş ve birbirleriyle kıyaslanmıştır.
---

## Mimari
```
cancer-screening-ml-project/
│
├── data/               # CDC BRFSS ham ve düzenlenmiş (.csv) veri setleri
├── notebooks/          # Keşifçi Veri Analizi (EDA) ve prototip modelleme Jupyter Notebook'ları
├── src/                # Model pipeline, veri ön işleme, eğitim ve tahmin Python (.py) betikleri
├── docs/               # Araştırma raporları, literatür taraması ve teknik mimari belgeleri
├── .gitignore          # Git takibinden dışlanacak gereksiz ve büyük dosyalar
├── .gitkeep            # Boş klasör yapılarının korunması için
├── README.md           # Proje tanıtımı
└── Extras
```
---
<div align="center">
  <img width="552" height="664" alt="ataturk" src="https://github.com/user-attachments/assets/78fbb9f4-3472-43f3-b809-440d3070ed84" />
  <p>"1881-193∞"</p>
</div>
