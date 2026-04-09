"""
SIPADMA — ML Engine
PDAM Cabang Semarang Timur
Metode: Z-Score + Rule-Based Scoring (6 bulan)
Kolom data asli: no_langganan, nama_pelanggan, alamat, no_telp, status,
                 golongan, stand_awal, stand_akhir, pemakaian, tagihan, bulan
"""
import pandas as pd
import numpy as np
import os

DATA_PATH = 'data/data_pdam.csv'

ZSCORE_TINGGI = 2.5
ZSCORE_SEDANG = 1.8
SPIKE_PCT     = 1.5
DROP_PCT      = 0.25
DELTA_RATIO   = 2.0

# ── Mapping nama bulan ke angka ────────────────────────────────────────────────
BULAN_MAP = {
    'januari':1,'februari':2,'maret':3,'april':4,'mei':5,'juni':6,
    'juli':7,'agustus':8,'september':9,'oktober':10,'november':11,'desember':12,
}

def parse_bulan(s):
    """Ubah 'September 2025' → bulan=9, tahun=2025"""
    s = str(s).strip().lower()
    for nm, num in BULAN_MAP.items():
        if nm in s:
            try:
                tahun = int(''.join(filter(str.isdigit, s)))
                return num, tahun
            except:
                return num, 2025
    return 1, 2025

def load_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    df  = pd.read_excel(filepath) if ext in ['.xlsx','.xls'] else pd.read_csv(filepath)

    # Normalisasi nama kolom
    df.columns = (df.columns.astype(str)
                  .str.strip().str.lower()
                  .str.replace(' ','_')
                  .str.replace('(','').str.replace(')','')
                  .str.replace('³','3').str.replace('rp','')
                  .str.strip('_'))

    # ── Deteksi kolom no_langganan ──
    for cand in ['no_langganan','no langganan','nomor_langganan','id_pelanggan']:
        if cand in df.columns:
            df = df.rename(columns={cand: 'no_langganan'})
            break

    # ── Deteksi kolom pemakaian / volume ──
    for cand in ['pemakaian_m3','pemakaian','volume_m3','volume','m3']:
        if cand in df.columns:
            df = df.rename(columns={cand: 'pemakaian'})
            break

    # ── Deteksi kolom tagihan ──
    for cand in ['tagihan_','tagihan','total_byr','total_bayar','bayar']:
        if cand in df.columns:
            df = df.rename(columns={cand: 'tagihan'})
            break

    # ── Deteksi kolom golongan ──
    for cand in ['golongan','golongan_tarif','trf','tarif']:
        if cand in df.columns:
            df = df.rename(columns={cand: 'golongan'})
            break

    # Validasi wajib
    required = ['no_langganan','pemakaian']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}. Kolom tersedia: {df.columns.tolist()}")

    # Bersihkan nilai
    df['no_langganan'] = df['no_langganan'].astype(str).str.lstrip("'").str.strip()
    df['pemakaian']    = pd.to_numeric(df['pemakaian'], errors='coerce').fillna(0).clip(lower=0)

    # ── Parse bulan/tahun dari kolom 'bulan' ──
    if 'bulan' in df.columns:
        parsed   = df['bulan'].apply(parse_bulan)
        df['bln'] = parsed.apply(lambda x: x[0])
        df['thn'] = parsed.apply(lambda x: x[1])
    else:
        df['bln'] = df.get('bulan_num', 1)
        df['thn'] = df.get('tahun', 2025)

    df['periode'] = pd.to_datetime(
        df['thn'].astype(str) + '-' + df['bln'].astype(str).str.zfill(2))

    # Kolom opsional
    for col, default in [('nama_pelanggan','-'),('alamat','-'),
                          ('no_telp','-'),('status','-'),
                          ('golongan','-'),('tagihan',0),
                          ('stand_awal',0),('stand_akhir',0),
                          ('bulan','-')]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default if default != 0 else 0)

    return df

def hitung_statistik(df):
    df = df.sort_values(['no_langganan','periode']).copy()
    g  = df.groupby('no_langganan')['pemakaian']

    df['mean_6bln']  = g.transform('mean')
    df['std_6bln']   = g.transform('std').fillna(0)
    df['std_6bln']   = df[['std_6bln','mean_6bln']].apply(
        lambda r: r['mean_6bln']*0.1 if r['std_6bln']==0 else r['std_6bln'], axis=1)
    df['median_6bln']= g.transform('median')
    df['delta']      = g.transform(lambda x: x.diff().fillna(0))
    df['pct_change'] = g.transform(
        lambda x: x.pct_change().fillna(0).replace([np.inf,-np.inf], 0))
    df['rasio_mean'] = (df['pemakaian'] / df['mean_6bln'].replace(0,1)).clip(0, 50)
    df['zscore']     = (df['pemakaian'] - df['mean_6bln']) / df['std_6bln']
    return df

def detect_anomaly(df):
    df = hitung_statistik(df)

    df['flag_zscore_tinggi'] = df['zscore'].abs() > ZSCORE_TINGGI
    df['flag_zscore_sedang'] = (df['zscore'].abs() > ZSCORE_SEDANG) & \
                               (df['zscore'].abs() <= ZSCORE_TINGGI)
    df['flag_spike']         = df['rasio_mean'] > (1 + SPIKE_PCT)
    df['flag_drop']          = (df['rasio_mean'] < DROP_PCT) & (df['pemakaian'] > 0)
    df['flag_delta']         = df['delta'].abs() > (df['std_6bln'] * DELTA_RATIO)

    df['skor'] = (
        df['flag_zscore_tinggi'].astype(int) * 3 +
        df['flag_zscore_sedang'].astype(int) * 2 +
        df['flag_spike'].astype(int)         * 2 +
        df['flag_drop'].astype(int)          * 2 +
        df['flag_delta'].astype(int)         * 1
    )
    df['is_anomali'] = df['skor'] >= 2

    def risiko(row):
        if row['skor'] >= 5 or abs(row['zscore']) > 4: return 'Tinggi'
        if row['skor'] >= 3 or abs(row['zscore']) > ZSCORE_TINGGI: return 'Sedang'
        if row['is_anomali']: return 'Rendah'
        return 'Normal'
    df['risiko'] = df.apply(risiko, axis=1)

    def buat_alasan(row):
        al = []
        z  = row['zscore']
        if z > ZSCORE_TINGGI:
            al.append(f"Lonjakan {z:.1f}σ di atas rata-rata ({row['pemakaian']:.1f} vs avg {row['mean_6bln']:.1f} m³)")
        elif z < -ZSCORE_TINGGI:
            al.append(f"Penurunan {abs(z):.1f}σ di bawah rata-rata ({row['pemakaian']:.1f} vs avg {row['mean_6bln']:.1f} m³)")
        if row['flag_spike']:
            al.append(f"Pemakaian {row['rasio_mean']:.1f}x rata-rata 6 bulan")
        if row['flag_drop']:
            al.append(f"Pemakaian turun ke {row['rasio_mean']*100:.0f}% dari rata-rata")
        if row['flag_delta'] and not row['flag_spike'] and not row['flag_drop']:
            arah = 'naik' if row['delta'] > 0 else 'turun'
            al.append(f"Perubahan mendadak {arah} {abs(row['pct_change'])*100:.0f}% dari bulan lalu")
        return '; '.join(al) if al else 'Pola tidak normal terdeteksi'
    df['alasan'] = df.apply(buat_alasan, axis=1)

    def jenis(row):
        if not row['is_anomali']: return 'Normal'
        if row['flag_spike'] or row['zscore'] > ZSCORE_TINGGI: return 'Lonjakan'
        if row['flag_drop']  or row['zscore'] < -ZSCORE_TINGGI: return 'Penurunan Drastis'
        return 'Tidak Wajar'
    df['jenis_anomali'] = df.apply(jenis, axis=1)

    return df

def get_summary(df):
    adf = df[df['is_anomali'] == True]
    return {
        'total_pelanggan':   int(df['no_langganan'].nunique()),
        'total_data':        int(len(df)),
        'total_anomali':     int(len(adf)),
        'pct_anomali':       round(len(adf)/max(len(df),1)*100, 1),
        'pelanggan_anomali': int(adf['no_langganan'].nunique()),
        'per_golongan': adf.groupby('golongan')['no_langganan'].count()
                           .sort_values(ascending=False).to_dict(),
        'per_risiko':   {str(k):int(v) for k,v in
                         adf['risiko'].value_counts().to_dict().items()},
        'per_bulan':    {str(k):int(v) for k,v in
                         adf.groupby('bulan')['no_langganan'].count()
                         .to_dict().items()},
        'per_jenis':    {str(k):int(v) for k,v in
                         adf['jenis_anomali'].value_counts().to_dict().items()},
    }

def update_data(new_filepath):
    df_new = load_data(new_filepath)
    if os.path.exists(DATA_PATH):
        df_old = load_data(DATA_PATH)
        df_all = pd.concat([df_old, df_new]).drop_duplicates(
            subset=['no_langganan','bln','thn'])
    else:
        df_all = df_new
    cols = ['no_langganan','nama_pelanggan','alamat','no_telp','status',
            'golongan','stand_awal','stand_akhir','pemakaian','tagihan','bulan']
    save_cols = [c for c in cols if c in df_all.columns]
    df_all[save_cols].to_csv(DATA_PATH, index=False)
    return detect_anomaly(df_all)
