import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Almacenamiento temporal de candidatos (en un entorno real, usar una base de datos)
candidatos = []

# Función para procesar texto
def procesar_texto(texto):
    stop_words = set(stopwords.words('spanish'))
    palabras = word_tokenize(texto)
    palabras_filtradas = [palabra for palabra in palabras if palabra.lower() not in stop_words and palabra.isalnum()]
    return ' '.join(palabras_filtradas)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apply', methods=['GET', 'POST'])
def apply():
    if request.method == 'POST':
        nombre = request.form['nombre']
        habilidades = request.form['habilidades']
        experiencia = int(request.form['experiencia'])

        # Manejar la carga del archivo
        cv_file = request.files['cv']
        cv_filename = f"{nombre}_{cv_file.filename}"
        cv_path = os.path.join(app.config['UPLOAD_FOLDER'], cv_filename)
        cv_file.save(cv_path)

        candidatos.append({
            'nombre': nombre,
            'cv': cv_filename,
            'habilidades': habilidades,
            'experiencia': experiencia
        })

        return redirect(url_for('index'))

    return render_template('apply.html')

@app.route('/admin', methods=['GET'])
def admin():
    if len(candidatos) == 0:
        return render_template('admin.html', candidatos=[], mejor_candidato=None)

    # Crear DataFrame con los datos de los candidatos
    candidatos_df = pd.DataFrame(candidatos)

    # Procesar CVs
    candidatos_df['cv_procesado'] = candidatos_df['habilidades'].apply(procesar_texto)

    # Transformar habilidades y experiencia en características
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(candidatos_df['cv_procesado'])
    y = candidatos_df['experiencia']

    # Entrenar un modelo
    modelo = LogisticRegression()
    modelo.fit(X, y)

    # Predecir la experiencia para nuevos candidatos
    predicciones = modelo.predict(X)

    # Seleccionar al mejor candidato basado en la predicción
    mejor_candidato_idx = predicciones.argmax()
    mejor_candidato = candidatos_df.iloc[mejor_candidato_idx]

    return render_template('admin.html', candidatos=candidatos, mejor_candidato=mejor_candidato)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
