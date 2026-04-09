"""
Generate data simulasi PDAM Cabang Semarang Timur
Kolom sesuai data asli: no_langganan, nama_pelanggan, alamat, no_telp,
                        status, golongan, stand_awal, stand_akhir,
                        pemakaian, tagihan, bulan
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

GOLONGAN = {'RT 3':(5,15),'RT 4':(8,22),'RT 5':(15,45),
            'Niaga 1':(20,60),'Niaga 2':(40,120),'Niaga 3':(80,200)}
GOL_PROB = [0.15,0.50,0.12,0.10,0.08,0.05]
BULAN_LIST = ['September 2025','Oktober 2025','November 2025',
              'Desember 2025','Januari 2026','Februari 2026']
NAMA_DEPAN   = ['Budi','Siti','Ahmad','Dewi','Agus','Rina','Hendra','Yuni',
                'Doni','Eka','Fajar','Lina','Rizky','Maya','Wahyu','Putri']
NAMA_BELAKANG= ['Santoso','Rahayu','Wijaya','Kusuma','Pratama','Sari',
                'Utama','Hidayat','Lestari','Nugroho','Susanto','Setiawan']
JALANAN = ['Kedondong','Salak','Cerme','Bayem','Lamper','Rogojembangan',
           'Kinibalu','Purwosari','Karang Kimpul','Medoho']

def generate_data(n=333):
    records, n_anom = [], 0
    for i in range(n):
        pid   = f"039{str(90000+i).zfill(5)}"
        nama  = f"{np.random.choice(NAMA_DEPAN)} {np.random.choice(NAMA_BELAKANG)}"
        jln   = np.random.choice(JALANAN)
        no    = np.random.randint(1,20)
        rt    = np.random.randint(1,8)
        rw    = np.random.randint(1,10)
        gol   = np.random.choice(list(GOLONGAN.keys()), p=GOL_PROB)
        lo,hi = GOLONGAN[gol]
        base  = np.random.uniform(lo, hi)
        stand = np.random.randint(1000, 9999)

        has_anom  = np.random.random() < 0.18
        anom_idx  = set()
        if has_anom:
            n_ev = np.random.randint(1, 3)
            anom_idx = set(np.random.choice(range(2,6), n_ev, replace=False))
            n_anom  += len(anom_idx)

        for idx, bulan in enumerate(BULAN_LIST):
            noise = np.random.normal(0, base*0.07)
            vol   = max(1, round(base + noise, 0))
            if idx in anom_idx:
                atype = np.random.choice(['spike','drop'], p=[0.65,0.35])
                vol   = round(vol * np.random.uniform(2.5,5.0), 0) if atype=='spike' \
                        else round(vol * np.random.uniform(0.03,0.18), 0)
            vol = int(vol)
            tagihan = vol * np.random.randint(4500, 7000)
            records.append({
                'no_langganan':   pid,
                'nama_pelanggan': nama,
                'alamat':         f"{jln} {no} Rt{rt}/{rw}",
                'no_telp':        f"3{np.random.randint(100000,999999)}",
                'status':         np.random.choice(['T','Y'], p=[0.85,0.15]),
                'golongan':       gol,
                'stand_awal':     stand,
                'stand_akhir':    stand + vol,
                'pemakaian':      vol,
                'tagihan':        tagihan,
                'bulan':          bulan,
            })
            stand += vol

    df = pd.DataFrame(records)
    os.makedirs('data', exist_ok=True)
    df.to_excel('data/data_pdam.xlsx', index=False)
    df.to_csv('data/data_pdam.csv', index=False)
    print("="*50)
    print("DATA SIMULASI PDAM SEMARANG TIMUR")
    print("="*50)
    print(f"  Pelanggan : {n}")
    print(f"  Periode   : Sep 2025 – Feb 2026 (6 bulan)")
    print(f"  Total baris: {len(df):,}")
    print(f"  Anomali sim: {n_anom}")
    print("="*50)
    return df

if __name__ == '__main__':
    generate_data()
