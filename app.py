from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os, json
import pandas as pd
from ml.detector import load_data, detect_anomaly, get_summary, update_data, DATA_PATH

app = Flask(__name__)
app.secret_key = 'sipadma-semarang-timur-2024'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_cache = {'df': None, 'summary': None}

def get_result(force=False):
    if _cache['df'] is None or force:
        if not os.path.exists(DATA_PATH):
            return None, None
        df = load_data(DATA_PATH)
        df = detect_anomaly(df)
        _cache['df']      = df
        _cache['summary'] = get_summary(df)
    return _cache['df'], _cache['summary']

@app.route('/')
def dashboard():
    df, summary = get_result()
    if df is None: return render_template('no_data.html')
    return render_template('dashboard.html', summary=summary)

@app.route('/anomali')
def anomali():
    df, summary = get_result()
    if df is None: return redirect(url_for('upload'))

    golongan = request.args.get('golongan','')
    risiko   = request.args.get('risiko','')
    jenis    = request.args.get('jenis','')
    bulan    = request.args.get('bulan','')
    page     = int(request.args.get('page',1))
    per_page = 50

    adf = df[df['is_anomali']==True].copy()
    if golongan: adf = adf[adf['golongan']      == golongan]
    if risiko:   adf = adf[adf['risiko']         == risiko]
    if jenis:    adf = adf[adf['jenis_anomali']  == jenis]
    if bulan:    adf = adf[adf['bulan'].astype(str) == bulan]

    adf   = adf.sort_values(['risiko','zscore'], ascending=[True,False])
    total = len(adf)
    adf   = adf.iloc[(page-1)*per_page:page*per_page]

    rows = []
    for _, r in adf.iterrows():
        rows.append({
            'no_langganan':   str(r['no_langganan']),
            'nama_pelanggan': str(r.get('nama_pelanggan','-')),
            'alamat':         str(r.get('alamat','-')),
            'golongan':       str(r.get('golongan','-')),
            'stand_awal':     r.get('stand_awal',0),
            'stand_akhir':    r.get('stand_akhir',0),
            'pemakaian':      round(float(r['pemakaian']),2),
            'tagihan':        r.get('tagihan',0),
            'bulan':          str(r.get('bulan','-')),
            'mean_6bln':      round(float(r['mean_6bln']),2),
            'zscore':         round(float(r['zscore']),2),
            'rasio_mean':     round(float(r['rasio_mean']),2),
            'risiko':         str(r['risiko']),
            'jenis_anomali':  str(r['jenis_anomali']),
            'alasan':         str(r['alasan']),
            'skor':           int(r['skor']),
        })

    filters = {
        'golongan_list': sorted(df['golongan'].dropna().unique().tolist()),
        'bulan_list':    df['bulan'].dropna().unique().tolist(),
        'jenis_list':    ['Lonjakan','Penurunan Drastis','Tidak Wajar'],
        'sel_golongan':  golongan,
        'sel_risiko':    risiko,
        'sel_jenis':     jenis,
        'sel_bulan':     bulan,
    }
    pagination = {'page':page,'per_page':per_page,'total':total,
                  'pages':max(1,(total+per_page-1)//per_page)}
    return render_template('anomali.html', rows=rows, summary=summary,
                           filters=filters, pagination=pagination)

@app.route('/pelanggan/<pid>')
def detail(pid):
    df, summary = get_result()
    if df is None: return redirect(url_for('upload'))
    pdata = df[df['no_langganan']==pid].sort_values('periode')
    if pdata.empty:
        flash('Pelanggan tidak ditemukan.','error')
        return redirect(url_for('anomali'))
    info  = pdata.iloc[-1]
    chart = {
        'labels':    pdata['bulan'].astype(str).tolist(),
        'pemakaian': [round(float(v),2) for v in pdata['pemakaian']],
        'mean':      [round(float(v),2) for v in pdata['mean_6bln']],
        'anomali':   pdata['is_anomali'].tolist(),
        'zscore':    [round(float(v),2) for v in pdata['zscore']],
        'tagihan':   [float(v) for v in pdata['tagihan']],
    }
    rows = []
    for _, r in pdata.iterrows():
        rows.append({
            'bulan':       str(r.get('bulan','-')),
            'stand_awal':  r.get('stand_awal',0),
            'stand_akhir': r.get('stand_akhir',0),
            'pemakaian':   round(float(r['pemakaian']),2),
            'tagihan':     r.get('tagihan',0),
            'mean_6bln':   round(float(r['mean_6bln']),2),
            'zscore':      round(float(r['zscore']),2),
            'rasio_mean':  round(float(r['rasio_mean']),2),
            'pct_change':  round(float(r['pct_change'])*100,1),
            'skor':        int(r['skor']),
            'is_anomali':  bool(r['is_anomali']),
            'risiko':      str(r['risiko']),
            'jenis_anomali':str(r['jenis_anomali']),
            'alasan':      str(r['alasan']),
        })
    return render_template('detail.html', pid=pid, info=info,
                           chart=json.dumps(chart), rows=rows, summary=summary)

@app.route('/pelanggan')
def semua_pelanggan():
    df, summary = get_result()
    if df is None: return redirect(url_for('upload'))

    search   = request.args.get('q','').strip().lower()
    golongan = request.args.get('golongan','')
    page     = int(request.args.get('page',1))
    per_page = 100

    grp = df.groupby('no_langganan').agg(
        nama_pelanggan=('nama_pelanggan','first'),
        alamat=('alamat','first'),
        golongan=('golongan','first'),
        total_anomali=('is_anomali','sum'),
        rata_pemakaian=('pemakaian','mean'),
        max_zscore=('zscore',lambda x: x.abs().max()),
        max_skor=('skor','max'),
    ).reset_index()

    if search:
        grp = grp[grp['nama_pelanggan'].str.lower().str.contains(search,na=False)|
                  grp['no_langganan'].str.lower().str.contains(search,na=False)]
    if golongan: grp = grp[grp['golongan']==golongan]

    grp   = grp.sort_values('total_anomali',ascending=False)
    total = len(grp)
    grp   = grp.iloc[(page-1)*per_page:page*per_page]

    rows = grp.to_dict('records')
    for r in rows:
        r['total_anomali']  = int(r['total_anomali'])
        r['rata_pemakaian'] = round(float(r['rata_pemakaian']),1)
        r['max_zscore']     = round(float(r['max_zscore']),2)
        r['max_skor']       = int(r['max_skor'])

    filters = {
        'golongan_list': sorted(df['golongan'].dropna().unique().tolist()),
        'sel_golongan':  golongan,
    }
    pagination = {'page':page,'per_page':per_page,'total':total,
                  'pages':max(1,(total+per_page-1)//per_page)}
    return render_template('pelanggan.html', rows=rows, summary=summary,
                           search=search, filters=filters, pagination=pagination)

@app.route('/upload', methods=['GET','POST'])
def upload():
    df, summary = get_result()
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename=='':
            flash('Pilih file terlebih dahulu.','error')
            return redirect(request.url)
        f = request.files['file']
        if not f.filename.lower().endswith(('.xlsx','.xls','.csv')):
            flash('Format harus .xlsx, .xls, atau .csv','error')
            return redirect(request.url)
        path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(path)
        try:
            result = update_data(path)
            _cache['df']      = result
            _cache['summary'] = get_summary(result)
            flash('Data berhasil diperbarui dan dianalisis ulang.','success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f'Gagal memproses file: {str(e)}','error')
            return redirect(request.url)
    return render_template('upload.html', summary=summary)

# ── API ─────────────────────────────────────────────────────────────────────────
@app.route('/api/chart/bulan')
def api_bulan():
    df, _ = get_result()
    if df is None: return jsonify({})
    t = df.groupby('bulan').agg(total=('pemakaian','count'),
                                anomali=('is_anomali','sum')).reset_index()
    return jsonify({'labels':t['bulan'].tolist(),
                    'total':t['total'].tolist(),
                    'anomali':t['anomali'].tolist()})

@app.route('/api/chart/golongan')
def api_golongan():
    _, s = get_result()
    if not s: return jsonify({})
    d = s['per_golongan']
    return jsonify({'labels':list(d.keys()),'values':list(d.values())})

@app.route('/api/chart/risiko')
def api_risiko():
    _, s = get_result()
    if not s: return jsonify({})
    d = s['per_risiko']
    return jsonify({'labels':list(d.keys()),'values':list(d.values())})

@app.route('/api/chart/jenis')
def api_jenis():
    _, s = get_result()
    if not s: return jsonify({})
    d = s['per_jenis']
    return jsonify({'labels':list(d.keys()),'values':list(d.values())})

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print("Jalankan dulu: python ml/generate_data.py")
    else:
        get_result()
    app.run(debug=True, port=5000)
