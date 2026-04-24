import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="Editor de Frames Preciso", layout="centered")
st.title("🎯 Selector de Frame de Precisión")

uploaded_file = st.file_uploader("Subí tu video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    temp_filename = f"temp_{os.getpid()}.mp4"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.read())

    # Procesamiento inicial para encontrar puntos de interés
    cap = cv2.VideoCapture(temp_filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Buscamos el cambio más brusco para centrar el slider ahí
    max_diff = 0
    pico_idx = 0
    ret, prev_frame = cap.read()
    if ret:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        for i in range(1, total_frames):
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = np.mean(cv2.absdiff(gray, prev_gray))
            if diff > max_diff:
                max_diff = diff
                pico_idx = i
            prev_gray = gray

    st.divider()
    st.subheader("Control de Precisión")
    
    # Slider para moverte frame a frame
    # Lo centramos en el 'pico_idx' que detectó el algoritmo
    current_frame_idx = st.slider("Mové el slider para buscar el frame exacto:", 
                                  0, total_frames - 1, int(pico_idx))

    # Leer y mostrar el frame seleccionado
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    ret, frame_final = cap.read()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame_final, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame actual: {current_frame_idx} | Tiempo: {current_frame_idx/fps:.2f}s", use_column_width=True)
        
        # Botón de descarga para el frame elegido manualmente
        res_pil = Image.fromarray(frame_rgb)
        res_pil.save("captura_manual.png")
        with open("captura_manual.png", "rb") as file:
            st.download_button("📥 Descargar este Frame", file, f"frame_{current_frame_idx}.png", "image/png")

    cap.release()
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
