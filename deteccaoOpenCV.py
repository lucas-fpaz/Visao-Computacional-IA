import cv2
import matplotlib.pyplot as plt

# Lê a imagem e prepara
imagem = cv2.imread('parking-825371_1920.webp')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
imagem_suave = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
bordas = cv2.Canny(imagem_suave, 50, 150)

# Encontra os contornos
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenha os contornos e caixas na imagem original
imagem_com_caixas = imagem.copy()
for contorno in contornos:
    x, y, w, h = cv2.boundingRect(contorno)
    cv2.rectangle(imagem_com_caixas, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Exibe imagem com objetos detectados
imagem_rgb = cv2.cvtColor(imagem_com_caixas, cv2.COLOR_BGR2RGB)
plt.imshow(imagem_rgb)
plt.title("Objetos Detectados com Contornos")
plt.axis('off')
plt.show()

# Conta os objetos detectados
print(f"Objetos detectados: {len(contornos)}")
