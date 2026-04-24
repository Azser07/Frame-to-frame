import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="Extractor Pro con Miniaturas", layout="wide")
st.title("🎯 Extractor de Precisión con Visualización")

if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0

uploaded_file = st.file_uploader("Subí tu video de Instagram", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    temp_filename = f"temp_{os.getpid()}.mp4"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 1. ANÁLISIS Y VISUALIZACIÓN DE RECOMENDADOS
    st.subheader("💡 Momentos clave detectados (Elegí uno)")
    picos = []
    ret, prev_frame = cap.read()
    if ret:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        for i in range(1, total_frames, 3): # Salto de 3 para mayor velocidad
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = np.mean(cv2.absdiff(gray, prev_gray))
            if diff > 12: 
                picos.append((i, frame.copy(), diff))
            prev_gray = gray

    # Ordenar por importancia y mostrar miniaturas
    picos.sort(key=lambda x: x[2], reverse=True)
    
    # Crear fila de miniaturas (mostramos las mejores 4)
    cols_reco = st.columns(4)
    for idx, (f_idx, f_img, score) in enumerate(picos[:4]):
        with cols_reco[idx]:
            img_mini = cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB)
            st.image(img_mini, use_column_width=True)
            if st.button(f"Seleccionar #{f_idx}", key=f"btn_{f_idx}"):
                st.session_state.current_frame = f_idx

    st.divider()

    # 2. PANEL DE NAVEGACIÓN MANUAL (Botonera)
    st.subheader("⚙️ Ajuste Fino Cuadro a Cuadro")
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    if c1.button("-10"): st.session_state.current_frame -= 10
    if c2.button("-5"):  st.session_state.current_frame -= 5
    if c3.button("-1"):  st.session_state.current_frame -= 1
    if c4.button("+1"):  st.session_state.current_frame += 1
    if c5.button("+5"):  st.session_state.current_frame += 5
    if c6.button("+10"): st.session_state.current_frame += 10

    st.session_state.current_frame = max(0, min(st.session_state.current_frame, total_frames - 1))

    # Mostrar Frame Actual en Grande
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
    ret, frame_final = cap.read()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame_final, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame actual: {st.session_state.current_frame}", use_column_width=True)
        
        # Descarga
        res_pil = Image.fromarray(frame_rgb)
        res_pil.save("captura.png")
        with open("captura.png", "rb") as file:
            st.download_button("📥 DESCARGAR FOTO FINAL", file, f"resultado_{st.session_state.current_frame}.png", "image/png")

    cap.release()
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
