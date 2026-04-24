import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="Extractor Rápido", layout="centered")
st.title("📸 Extractor de Fotogramas Críticos")
st.write("Analizá grabaciones de Instagram y limpiá el temporal automáticamente.")

uploaded_file = st.file_uploader("Subí tu grabación de pantalla", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 1. Creamos un nombre temporal único
    temp_filename = f"temp_video_{os.getpid()}.mp4"
    
    with st.spinner("Analizando video cuadro por cuadro..."):
        # Guardamos el archivo para que OpenCV pueda procesarlo
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_filename)
        max_diff = 0
        best_frame = None
        
        ret, prev_frame = cap.read()
        if ret:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesamiento rápido en escala de grises
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, prev_gray)
                score = np.sum(diff)

                if score > max_diff:
                    max_diff = score
                    best_frame = frame.copy() # Guardamos copia del mejor frame
                
                prev_gray = gray

        cap.release()
        
        # 2. ACCIÓN CRÍTICA: Eliminamos el video del servidor inmediatamente
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    # 3. Mostrar resultados
    if best_frame is not None:
        st.success("✅ Análisis completo. El archivo de video fue eliminado del servidor.")
        
        # Convertir para mostrar en web
        result_img = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption="Fotograma con el cambio más rápido detectado", use_column_width=True)
        
        # Opción de descarga para el usuario
        res_pil = Image.fromarray(result_img)
        res_pil.save("captura_detectada.png")
        with open("captura_detectada.png", "rb") as file:
            st.download_button("Descargar Imagen", file, "foto_detectada.png", "image/png")
    else:
        st.error("No se pudo procesar el video.")
