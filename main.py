from flask import Flask, request, redirect, url_for, render_template, jsonify, session
from werkzeug.utils import secure_filename
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
import pandas as pd
import os
import pickle
import numpy as np

kolom_angka = ['luas tanah (m2)', 'luas bangunan (m2)', 'jarak pusat kota (km)', 'usia bangunan (tahun)', 'jumlah lantai', 'jumlah kamar', 'harga (juta)']
kolom_kategori = ['akses jalan']
standarisasi = StandardScaler()
label_encoders = {}

app = Flask(__name__)
app.secret_key = 'machine_learning'
if not os.path.exists("dataset"):
  os.makedirs("dataset")

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file ():
  file = request.files['file']
  if file and file.filename.endswith('.csv'):
    form = request.form
    filepath = os.path.join("dataset", secure_filename(file.filename))
    file.save(filepath)
    session ['filepath'] = filepath 

    data = pd.read_csv(filepath)
    data = data[(kolom_angka+kolom_kategori)+['zona lingkungan']]

    #Transformasi Data
    for col in kolom_kategori:
      le = LabelEncoder()
      data[col] = le.fit_transform(data[col])
      label_encoders[col] = le
    data[kolom_angka] = standarisasi.fit_transform(data[kolom_angka])

    input = data.drop(columns='zona lingkungan')
    label = data['zona lingkungan']

    input_train, input_test, label_train, label_test = \
      train_test_split(input, label, test_size=float(form['data_uji']), random_state=42)
    
    data.to_csv('data_hasil_uji.csv', index=False)
    
    #penentuan model
    model = SVC(C=float(form['C']),
                kernel=form['kernel'],
                gamma=float(form['y']) if form['y'] else 'scale',
                degree=int(form['d']),
                coef0=float(form['coef']))
    
    #Melatih model SVM
    model.fit(input_train, label_train)
    with open('main.pkl', 'wb') as f:
      pickle.dump(model, f)

    #Evaaluasi model
    y_pred = model.predict(input_test)
    accuracy = accuracy_score(label_test, y_pred) * 100
    precision = precision_score(label_test, y_pred, average='macro', zero_division=0) * 100
    recall = recall_score(label_test, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(label_test, y_pred, average='macro', zero_division=0) * 100
    data_uji_persen = str(round(float(form['data_uji']) * 100)) + " %"

    # Simpan hasil evaluasi di sesi
    session['hasil_evaluasi'] = {
      'data_uji' : data_uji_persen,
      'kernel' : str(form['kernel']),
      'parameter_regulasi' : str(form['C']),
      'parameter_gamma' : str(form['y']),
      'parameter_derajat' : str(form['d']),
      'parameter_coefisien' : str(form['coef']),
      'accuracy' : str(round(accuracy,2))+" %",
      'precision': str(round(precision))+" %",
      'recall': str(round(recall))+ " %",
      'f1_score': str(round(f1))+" %"
    }
    #Mengalihkan Halaman ke
    return redirect(url_for('evaluate'))
  return "Error: Pastikan file yang diunggah berformat CSV"

@app.route('/evaluate')
def evaluate():
  evaluation = session.get('hasil_evaluasi')
  return render_template('evaluasi.html', evaluation=evaluation)

@app.route('/predict')
def predict_page():
  return render_template('prediksi.html')

@app.route('/predict', methods=['POST'])
def predict():
  # Proses input untuk prediksi harga
  if 'filepath' in session:
    filepath = session['filepath']
    data = pd.read_csv(filepath)

    form = request.form

    # Ekstraksi fitur dan target
    Fitur = data[['luas tanah (m2)', 'luas bangunan (m2)', 'usia bangunan (tahun)','jarak pusat kota (km)' ]]
    Target = data['harga (juta)']

    # Membagi dua data ke train dan test
    Fitur_train, Fitur_test, Target_train, Target_test = train_test_split(
        Fitur, Target, test_size=float((form['data_uji'])), random_state=42)

    # Melatih data model
    model = SVR(kernel='poly', C=float(form['coef0']), degree=3)
    model.fit(Fitur_train, Target_train)

    # model evaluasi
    Target_pred = model.predict(Fitur_test)
    r2 = r2_score(Target_test, Target_pred)

    # Menginput data yang diprediksi
    luas_tanah = float(form['m2'])
    luas_bangunan = float(form['M2'])
    jarak_pusat_kota = float(form['km'])
    usia_bangunan = float(form['tahun'])
    input_data = pd.DataFrame(
        [[luas_tanah, luas_bangunan, jarak_pusat_kota, usia_bangunan]],
        columns=['luas tanah (m2)', 'luas bangunan (m2)', 'usia bangunan (tahun)', 'jarak pusat kota (km)']
    )

    data_uji_persen = str(round(float(form['data_uji']) * 100)) + " %"

    hasil_prediksi = model.predict(input_data)

    # Menyimpan hasil ke SESSION
    session['prediksi'] = {
        'data_uji' : data_uji_persen,
        'koefisien': (r2),
        'k_input' : round(r2,1),
        'harga': round(hasil_prediksi[0]),
        'luas_tanah' : str(form['m2']),
        'luas_bangunan' : str(form['M2']),
        'jarak_ke_kota' : str(form['km']),
        'usia_bangunan' : str(form['tahun'])
    }
    # Mengambil hasil session untuk di pindiahkan ke hasil prediksi
    return render_template('hasil_prediksi.html', prediksi=session['prediksi'])
  else:
    return "error: Tidak ada file yang diunggah sebelumnya untuk digunakan"


if __name__ == '__main__':
  app.run(port=5002, debug=True)