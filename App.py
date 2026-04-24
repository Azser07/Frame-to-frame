import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="Detector de Frames Flash", layout="centered")
st.title("📸 Capturador de Frames Ocultos")
st.write("Ideal para detectar el frame de foto entre videos o cuentas regresivas.")

uploaded_file = st.file_uploader("Subí tu grabación de pantalla", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    temp_filename = f"temp_video_{os.getpid()}.mp4"
    
    with st.spinner("Analizando transiciones rápidas..."):
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_filename)
        frames_sospechosos = []
        scores = []
        
        ret, prev_frame = cap.read()
        if ret:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Calculamos la diferencia con el anterior
                diff = cv2.absdiff(gray, prev_gray)
                score = np.mean(diff) # Usamos el promedio para suavizar ruido
                scores.append(score)

                # Guardamos frames que tengan un cambio significativo
                # Este es el 'trigger' para frames que duran poco
                if score > 15: # Umbral de sensibilidad (ajustable)
                    frames_sospechosos.append((frame_idx, frame.copy(), score))
                
                prev_gray = gray
                frame_idx += 1

        cap.release()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    # --- RESULTADOS ---
    if frames_sospechosos:
        st.success(f"Se detectaron {len(frames_sospechosos)} cambios rápidos.")
        
        # Ordenamos por los cambios más fuertes
        frames_sospechosos.sort(key=lambda x: x[2], reverse=True)
        
        st.write("### Posibles capturas encontradas:")
        # Mostramos los 3 mejores candidatos para que vos elijas el correcto
        cols = st.columns(2)
        for i, (idx, f, s) in enumerate(frames_sospechosos[:4]):
            with cols[i % 2]:
                img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=f"Candidato {i+1} (Frame {idx})")
                
                # Botón de descarga individual
                res_pil = Image.fromarray(img_rgb)
                res_pil.save(f"frame_{idx}.png")
                with open(f"frame_{idx}.png", "rb") as file:
                    st.download_button(f"Descargar Candidato {i+1}", file, f"foto_{idx}.png", "image/png")
    else:
        st.error("No se detectaron cambios bruscos. Intentá con un video más claro.")
