import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv('X_train_hazir.csv')
X_test  = pd.read_csv('X_test_hazir.csv')
y_train = pd.read_csv('y_train_hazir.csv').squeeze()
y_test  = pd.read_csv('y_test_hazir.csv').squeeze()


# MODELİ İSKELETİ. !!!WEİGHT BALANCİNG!!!
# class_weight='balanced'  kanser olanlar az olduğu için azınlık sınıfına daha fazla ağırlık vermeli ki model hatalı olmasın.
# max_iter=1000  modelin parametreleri bulması için max iterasyon sayısını ifade eder.
# ─────────────────────────────────────────────
model_lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# HİPERPARAMETRE OPTİMİZASYONU — GRID SEARCH
# C parametresi: ne kadar regularization uygulanacak?
# Küçük C = çok regularization = daha basit model
# Büyük C = az regularization = daha karmaşık model
# cv=5: veriyi 5 parçaya böl, 5 kez eğit/test et, ortala
# scoring='f1': en iyi C'yi F1 skoruna göre seç

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(
    model_lr,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Grid Search basliyor...")
grid.fit(X_train_scaled, y_train)

print(f"En iyi C degeri : {grid.best_params_['C']}")
print(f"En iyi F1 skoru : {grid.best_score_:.3f}")

# TAHMİN OLUŞTURMA
# predict  0 veya 1 (evet/hayır) / (kanser var mı yok mu kontrol eder.)
# predict_proba  olasılık. (AUC hesabi icin gerekli)!!!

best_lr = grid.best_estimator_
y_pred  = best_lr.predict(X_test_scaled)
y_prob  = best_lr.predict_proba(X_test_scaled)[:, 1]

# SONUÇLARI KAYDETTİK. BU YENİ DOSYALAR ANALİZ UZMANLARI TARAFINDAN KULLANILACAK.

np.save('y_pred_lr.npy', y_pred)
np.save('y_prob_lr.npy', y_prob)

print("Tahminler kaydedildi: y_pred_lr.npy, y_prob_lr.npy")
print("Lojistik Regresyon basaraili.")
