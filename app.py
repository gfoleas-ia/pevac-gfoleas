import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from pathlib import Path

# =============================
# CONFIGURACIÓN GENERAL
# =============================
st.set_page_config(
    page_title="SOPP+IA Sofía - Pie Equino Varo",
    page_icon="🦶",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "assets" / "soppia_pie_logo.png"
DEMO_PATH = BASE_DIR / "assets" / "pie_equino_varo.png"

# =============================
# ESTILO SOPP+IA CON TONO VERDE
# =============================
st.markdown("""
<style>
:root {
    --soppia-green: #006E4E;
    --soppia-dark-green: #004D36;
    --soppia-light-green: #E7F6EF;
    --soppia-orange: #F26A21;
    --soppia-text: #1E293B;
}

.stApp { background-color: #FFFFFF; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #006E4E 0%, #004D36 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    color: var(--soppia-dark-green);
    font-size: 42px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 8px;
}

.subtitle {
    color: var(--soppia-text);
    font-size: 18px;
    line-height: 1.6;
}

.warning-box {
    background-color: var(--soppia-light-green);
    border-left: 8px solid var(--soppia-orange);
    padding: 20px;
    border-radius: 12px;
    font-weight: 700;
    color: var(--soppia-dark-green);
    margin-top: 22px;
    margin-bottom: 22px;
}

.dev-box {
    background-color: #F0FAF5;
    border: 1px solid #B7E4CF;
    padding: 22px;
    border-radius: 14px;
    color: var(--soppia-text);
    margin-bottom: 18px;
}

.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #CFEBDD;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 3px 8px rgba(0,0,0,0.05);
}

.metric-number {
    color: var(--soppia-dark-green);
    font-size: 28px;
    font-weight: 800;
}

.metric-label {
    color: #334155;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# BARRA LATERAL
# =============================
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)
    st.markdown("## SOPP+IA Sofía - Pie Equino Varo")
    st.markdown("**Entrenamiento educativo en Ortopedia Pediátrica**")
    st.divider()
    st.markdown("### Para estudiantes, padres y cuidadores")
    st.write("Aplicación académica para explorar el uso de IA generativa en la descripción de imágenes no identificables de pies infantiles.")
    st.divider()
    st.markdown("**Desarrollado por:**")
    st.write("Dr. Geovanny F. Oleas-Santillán")
    st.write("Ortopedista Pediatra")
    st.write("Quito, Ecuador")

# =============================
# ENCABEZADO
# =============================
col1, col2 = st.columns([2.1, 1])

with col1:
    st.markdown("""
    <div class="main-title">
    SOPP+IA Sofía - Pie Equino Varo:<br>
    Descripción automática de imágenes de pies infantiles
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle">
    Aplicación académica de entrenamiento para explorar el uso de un modelo de IA
    en la generación de descripciones textuales simples a partir de imágenes no identificables.
    </div>
    """, unsafe_allow_html=True)

with col2:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), caption="SOPP+IA Sofía - Pie Equino Varo", use_container_width=True)

st.markdown("""
<div class="warning-box">
Esta aplicación es educativa y de entrenamiento. No diagnostica pie equino varo, no calcula la escala de Pirani y no reemplaza el examen físico, la valoración clínica ni el criterio del ortopedista pediatra.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dev-box">
<b>Modelo seleccionado:</b> Salesforce/blip-image-captioning-base<br>
<b>Objetivo:</b> generar una descripción textual simple de imágenes no identificables de pies infantiles.<br>
<b>Alcance:</b> apoyo descriptivo básico para aprendizaje; no es una herramienta diagnóstica.
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-card"><div class="metric-number">BLIP</div><div class="metric-label">Modelo IA Generativa</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-number">Image</div><div class="metric-label">Entrada visual</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-number">Text</div><div class="metric-label">Salida descriptiva</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-number">0</div><div class="metric-label">Diagnóstico clínico</div></div>', unsafe_allow_html=True)

# =============================
# CARGA DEL MODELO
# =============================
@st.cache_resource
def cargar_modelo():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

with st.spinner("Cargando modelo de Hugging Face..."):
    processor, model = cargar_modelo()

# =============================
# FUNCIÓN DE DESCRIPCIÓN
# =============================
def describir_imagen(imagen_pil):
    imagen_pil = imagen_pil.convert("RGB")
    inputs = processor(imagen_pil, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
    descripcion = processor.decode(output[0], skip_special_tokens=True)
    return descripcion

# =============================
# INTERFAZ PRINCIPAL
# =============================
st.header("Prueba funcional del modelo")
st.write("Suba una imagen no identificable de pies infantiles o use la imagen de ejemplo incluida.")

opcion = st.radio("Seleccione una opción:", ["Usar imagen de ejemplo", "Subir mi propia imagen"], horizontal=True)

imagen = None
if opcion == "Usar imagen de ejemplo":
    if DEMO_PATH.exists():
        imagen = Image.open(DEMO_PATH).convert("RGB")
    else:
        st.error("No se encontró la imagen de ejemplo.")
else:
    archivo = st.file_uploader("Subir imagen no identificable", type=["jpg", "jpeg", "png"])
    if archivo is not None:
        imagen = Image.open(archivo).convert("RGB")

if imagen is not None:
    col_img, col_res = st.columns([1, 1])
    with col_img:
        st.image(imagen, caption="Imagen analizada", use_container_width=True)

    with col_res:
        if st.button("Generar descripción automática"):
            descripcion = describir_imagen(imagen)
            st.subheader("Descripción automática generada por el modelo")
            st.success(descripcion)
            st.subheader("Interpretación educativa")
            st.write("""
            El modelo genera una frase descriptiva general de la imagen. En este proyecto,
            el resultado se usa como demostración educativa de inteligencia artificial aplicada
            a imágenes en ortopedia pediátrica.
            """)
            st.info("La interpretación clínica corresponde siempre al profesional de salud. El modelo no diagnostica, no mide deformidad y no calcula la escala de Pirani.")

st.divider()
st.subheader("Doble chequeo del alcance")
st.write("✅ Modelo de IA cargado correctamente.")
st.write("✅ Aplicación válida de inteligencia artificial: imagen a texto.")
st.write("✅ Ortopedia pediátrica: pie equino varo.")
st.write("✅ Uso educativo y de entrenamiento claramente declarado.")
st.write("✅ No se presenta como diagnóstico ni como cálculo de escala clínica.")
