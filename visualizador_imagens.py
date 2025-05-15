import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.title("Ponderada - Visualizador de imagens")

uploaded_file = st.file_uploader("Escolha a imgem", type=["png", "jpg", "jpeg"])

def load_image(upload):
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

def aplicar_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def aplicar_colormap(img, tipo):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(img_gray, tipo)

def aplicar_filtros(img, grayscale, inversao, contraste_val, blur_val, sharpen_val, bordas, angulo_rotacao, nova_largura, nova_altura, filtro_cor):
    imagem = img.copy()

    if grayscale:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

    if inversao:
        imagem = cv2.bitwise_not(imagem)

    if contraste_val > 1.0:
        imagem = cv2.convertScaleAbs(imagem, alpha=contraste_val, beta=0)

    if blur_val > 1:
        if blur_val % 2 == 0:
            blur_val += 1 
        imagem = cv2.GaussianBlur(imagem, (blur_val, blur_val), 0)

    if sharpen_val > 0:
        kernel = np.array([[0, -1, 0],
                           [-1, 5 + sharpen_val, -1],
                           [0, -1, 0]])
        imagem = cv2.filter2D(imagem, -1, kernel)

    if bordas:
        edges = cv2.Canny(imagem, 100, 200)
        imagem = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if angulo_rotacao != 0:
        (h, w) = imagem.shape[:2]
        centro = (w // 2, h // 2)
        matriz = cv2.getRotationMatrix2D(centro, angulo_rotacao, 1.0)
        imagem = cv2.warpAffine(imagem, matriz, (w, h))

    if nova_largura > 0 and nova_altura > 0:
        imagem = cv2.resize(imagem, (nova_largura, nova_altura))

    if filtro_cor == "Sepia":
        imagem = aplicar_sepia(imagem)
    elif filtro_cor == "Colormap - JET":
        imagem = aplicar_colormap(imagem, cv2.COLORMAP_JET)
    elif filtro_cor == "Colormap - OCEAN":
        imagem = aplicar_colormap(imagem, cv2.COLORMAP_OCEAN)
    elif filtro_cor == "Colormap - HOT":
        imagem = aplicar_colormap(imagem, cv2.COLORMAP_HOT)
    elif filtro_cor == "Colormap - RAINBOW":
        imagem = aplicar_colormap(imagem, cv2.COLORMAP_RAINBOW)

    return imagem

def convert_cv2_to_pil(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def image_to_bytes(img_pil):
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

if uploaded_file:
    imagem_cv2 = load_image(uploaded_file)

    st.subheader("Filtros comuns")
    col1, col2, col3 = st.columns(3)
    with col1:
        grayscale = st.checkbox("Cinza")
        inversao = st.checkbox("Inversão de cores")
    with col2:
        contraste_val = st.slider("Contraste", 1.0, 3.0, 1.0, step=0.1)
        blur_val = st.slider("Blur", 1, 21, 1, step=2)  # só ímpares
    with col3:
        sharpen_val = st.slider("Nitidez", 0.0, 5.0, 0.0, step=0.5)
        bordas = st.checkbox("Detecção de bordas")

    st.subheader("Rotação e redimensionamento")
    col4, col5 = st.columns(2)
    with col4:
        angulo_rotacao = st.slider("Rotação (graus)", -180, 180, 0)
    with col5:
        nova_largura = st.number_input("Nova largura (px)", min_value=1, value=imagem_cv2.shape[1])
        nova_altura = st.number_input("Nova altura (px)", min_value=1, value=imagem_cv2.shape[0])

    st.subheader("Filtros de Cor")
    filtro_cor = st.selectbox("Escolha um filtro de cor", [
        "Nenhum",
        "Sepia",
        "Colormap - JET",
        "Colormap - OCEAN",
        "Colormap - HOT",
        "Colormap - RAINBOW"
    ])

    imagem_editada = aplicar_filtros(
        imagem_cv2, grayscale, inversao,
        contraste_val, blur_val, sharpen_val,
        bordas, angulo_rotacao,
        nova_largura, nova_altura,
        filtro_cor
    )

    original = convert_cv2_to_pil(imagem_cv2)
    editada = convert_cv2_to_pil(imagem_editada)

    st.subheader("Visualização")
    col1, col2 = st.columns(2)
    col1.image(original, caption="Original", use_column_width=True)
    col2.image(editada, caption="Editada", use_column_width=True)

    st.download_button("Baixar foto final", data=image_to_bytes(editada),
                       file_name="foto_final.png", mime="image/png")
