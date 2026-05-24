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

## 📌 Proje Hakkında

Bu proje, kanser taraması, risk faktörlerinin tespiti ve erken teşhis süreçlerini optimize etmek amacıyla geliştirilmiş bir makine öğrenmesi araştırmasıdır. Hastaların demografik bilgileri, yaşam tarzı alışkanlıkları ve genel sağlık durumlarına ilişkin veriler analiz edilerek, bireylerin kanser riski taşıyıp taşımadığı veya kanser tarama programlarına uyum gösterip göstermeyeceği tahmin edilmektedir.

Üç farklı üniversitenin bilgi birikimi ve işbirliği ile disiplinlerarası bir akademik çalışma olarak hayata geçirilen bu proje, koruyucu hekimlik politikaları için veri odaklı bir karar destek mekanizması sunmayı hedeflemektedir.

## 📊 Veri Seti: 2024 BRFSS (SAS Transport Format)

Çalışmamızda **CDC (Centers for Disease Control and Prevention)** tarafından yayınlanan [2024 BRFSS (Behavioral Risk Factor Surveillance System)](https://www.cdc.gov/brfss/index.html) veri seti kullanılmıştır. 

* **Format:** Orijinal veri seti `SAS Transport Format (.xpt)` uzantılıdır.
* **İçerik:** Kanser öyküsü (cilt kanseri ve diğer kanser türleri), sigara/alkol tüketimi, fiziksel aktivite, kronik hastalık geçmişi ve sosyodemografik anket cevapları.
* **Veri Boyutu:** Veri seti oldukça büyük hacimli olduğundan (orijinal .xpt ve işlenmiş .csv dosyaları), GitHub reposunda barındırılmamaktadır (Bkz. *Kurulum*).

## 🔬 Metodoloji

Tabular anket verilerinden anlamlı sonuçlar üretebilmek için aşağıdaki veri bilimi boru hattı (pipeline) izlenmiştir:

1. **Veri Okuma ve Dönüştürme:** SAS `.xpt` formatındaki verilerin Python ortamına (`pandas.read_sas` kullanılarak) aktarılması ve Pandas DataFrame formatına dönüştürülmesi.
2. **Veri Ön İşleme (Data Preprocessing):** 
   * BRFSS özelindeki "Bilmiyorum/Reddedildi" (örn: 77, 99 kodlu) yanıtların eksik veri (NaN) olarak ele alınması.
   * Hedef değişkenin (kanser tanısı / tarama durumu) belirlenmesi ve sınıf dengesizliklerinin (SMOTE vb. yöntemlerle) giderilmesi.
3. **Özellik Mühendisliği (Feature Engineering):** Kanser teşhisi ile en çok korelasyon gösteren demografik ve davranışsal özelliklerin (Örn: `_AGEG5YR`, `SMOKE100`, `CHCSCNCR`) seçilmesi.
4. **Modelleme:** Tabular verilerde yüksek performans gösteren makine öğrenmesi algoritmalarının eğitilmesi:
  
5. **Değerlendirme (Evaluation):** Modeller; Doğruluk (Accuracy), Hassasiyet (Precision), Duyarlılık (Recall), F1-Skoru ve ROC-AUC metrikleri ile istatistiksel olarak ölçülmüş ve birbirleriyle kıyaslanmıştır.

---

## ⚙️ Kurulum ve Kullanım (Ekip İçi Rehber)

Veri dosyaları `.gitignore` ile korunduğu için GitHub'da yer almamaktadır. Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla izleyin:

**1. Repoyu Klonlayın**
Terminalinizi (CMD/Bash) açın ve projeyi bilgisayarınıza indirin:

*git clone [https://github.com/Grup7-Proje/cancer-screening-ml-project.git](https://github.com/Grup7-Proje/cancer-screening-ml-project.git)*
*cd cancer-screening-ml-project*

**2. Veri Setlerini İndirin ve Yerleştirin**

Ortak iletişim kanalımızda sabitlenen Google Drive bağlantısına gidin.

İndirdiğiniz ham .xpt dosyasını data/raw/ klasörünün içine yerleştirin.

İndirdiğiniz temizlenmiş .csv dosyalarını (X_train_hazir.csv vb.) data/processed/ klasörünün içine yerleştirin.

**3. Gerekli Kütüphaneleri Yükleyin**
Projenin çalışması için temel veri bilimi kütüphanelerine ihtiyacınız vardır

*pip install pandas numpy scikit-learn*

4. Modelleri Çalıştırın
Tüm veriler klasörlerde hazır olduğuna göre, src/ klasörü altındaki çalışma dosyalarını çalıştırarak eğitime ve testlere başlayabilirsiniz
---
<div align="center">
  <img width="736" height="887" alt="ataturk" src="https://github.com/user-attachments/assets/78fbb9f4-3472-43f3-b809-440d3070ed84" />
  <p>"1881-193∞"</p>
</div>
