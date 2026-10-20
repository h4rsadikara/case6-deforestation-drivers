# Studi Kasus 6: Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI

## Deskripsi
Klasifikasi penyebab langsung alert deforestasi (pertanian kecil, perkebunan, pembalakan/jalan, tambang, kebakaran) dengan Random Forest pada fitur turunan citra satelit. Fitur sintetis, sehingga akurasi sekitar 99% terlalu optimistis karena kelas sengaja dibuat terpisah; pada data nyata performa lebih rendah. Pengembangan: fitur nyata dari Sentinel-2/Landsat dan alert GLAD/RADD dari Global Forest Watch.

## Cara menjalankan
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Grafik tersimpan di folder `outputs/`.

## Catatan
Data pada proyek ini sintetis atau bawaan library agar bisa dijalankan tanpa dataset eksternal. Kesimpulan substantif perlu divalidasi dengan data nyata.
