# SOPP+IA Sofía - Descripción automática de imágenes de pies infantiles en pie equino varo

## S4-Taller: Implementación de un modelo de Hugging Face en Google Colab

### Título
Descripción automática de imágenes de pies infantiles en pie equino varo

### Datos personales
**Desarrollador:** Geovanny Fabricio Oleas Santillán  
**Profesión:** Ortopedista Pediatra  
**Lugar:** Quito, Ecuador

### Modelo seleccionado
`Salesforce/blip-image-captioning-base`

### Objetivo
Usar Hugging Face para generar una descripción textual simple de imágenes clínicas no identificables de pies infantiles.

### Alcance
Este modelo recibe una imagen y genera una frase descriptiva general.  
No diagnostica, no calcula la escala de Pirani y no reemplaza la valoración médica.

### Propósito educativo
Esta aplicación pertenece a la línea académica **SOPP+IA Sofía** y se presenta como una herramienta de entrenamiento para comprender cómo un modelo de visión-lenguaje puede describir imágenes.

### Archivos del repositorio
- `app.py`: aplicación Streamlit con estilo SOPP+IA en tono verde para pie equino varo.
- `requirements.txt`: librerías necesarias para ejecutar la aplicación.
- `assets/pie_equino_varo.png`: imagen de ejemplo.
- `assets/soppia_pie_logo.png`: imagen circular para la interfaz.
- `S4_Taller_HuggingFace_BLIP_Pie_Equino_Varo.ipynb`: notebook para Google Colab.

### Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Publicación en Streamlit Community Cloud
1. Crear un repositorio en GitHub.
2. Subir todos los archivos de esta carpeta.
3. Entrar a Streamlit Community Cloud.
4. Seleccionar el repositorio.
5. Indicar `app.py` como archivo principal.
6. Deploy.

### Advertencia
La aplicación es educativa y de entrenamiento. No realiza diagnóstico clínico, no identifica gravedad, no calcula Pirani y no reemplaza al ortopedista pediatra.
