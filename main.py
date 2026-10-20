"""Studi Kasus 6 - Attributing direct drivers to deforestation alerts with satellite imagery and AI.
Kasus: tiap alert deforestasi (mis. GLAD/RADD) diklasifikasikan penyebab langsungnya.
Di dunia nyata fitur diambil dari citra satelit (Sentinel-2/Landsat) + data spasial; di sini
fitur dibuat sintetis agar repo bisa dijalankan tanpa akses data. Pipeline-nya sama:
fitur per alert -> Random Forest -> evaluasi per kelas -> matriks kebingungan.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

rng = np.random.default_rng(11)
# kelas: (luas_ha, jarak_jalan_km, jarak_desa_km, dNDVI, kemiringan, api(0-1), keteraturan_bentuk(0-1))
profil = {
    "Pertanian kecil":  (1.5, 1.0, 1.0, -0.35, 8, 0.1, 0.3),
    "Perkebunan besar": (60, 3.0, 6.0, -0.55, 4, 0.1, 0.9),
    "Pembalakan/jalan": (8, 0.5, 8.0, -0.25, 15, 0.0, 0.2),
    "Tambang":          (20, 4.0, 10.0, -0.60, 12, 0.0, 0.5),
    "Kebakaran":        (35, 2.0, 4.0, -0.45, 6, 0.9, 0.1),
}
cols = ["luas_ha", "jarak_jalan_km", "jarak_desa_km", "delta_ndvi", "kemiringan", "indeks_api", "keteraturan_bentuk"]
rows = []
for name, mu in profil.items():
    n = 400
    X = np.column_stack([rng.lognormal(np.log(mu[0]), 0.5, n), rng.gamma(4, mu[1]/4, n), rng.gamma(4, mu[2]/4, n),
                         rng.normal(mu[3], 0.12, n), np.abs(rng.normal(mu[4], 4, n)),
                         np.clip(rng.normal(mu[5], 0.2, n), 0, 1), np.clip(rng.normal(mu[6], 0.2, n), 0, 1)])
    rows.append(pd.DataFrame(X, columns=cols).assign(penyebab=name))
df = pd.concat(rows, ignore_index=True)

X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df.penyebab, test_size=0.25, stratify=df.penyebab, random_state=0)
rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_tr, y_tr)
pred = rf.predict(X_te)
print(classification_report(y_te, pred))
print("Fitur terpenting:")
print(pd.Series(rf.feature_importances_, cols).sort_values(ascending=False).round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(y_te, pred, xticks_rotation=35, ax=ax, colorbar=False)
ax.set_title("Atribusi penyebab deforestasi (data sintetis)")
fig.tight_layout(); fig.savefig("outputs/case6_drivers.png", dpi=150)
print("Gambar tersimpan: outputs/case6_drivers.png")
