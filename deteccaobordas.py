import cv2
import matplotlib.pyplot as plt

# Lê a imagem e converte para escala de cinza
imagem = cv2.imread('Pinheiros_Avenida-Faria-Lima_Rua-Butanta_Foto-Marcos-Santos_U0Y1378-scaled.jpg')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplica um blur para reduzir ruído
imagem_suave = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)

# Detecta bordas com Canny
bordas = cv2.Canny(imagem_suave, 50, 150)

# Exibe resultados
titulos = ['Original (Cinza)', 'Com Blur', 'Bordas Canny']
imagens = [imagem_cinza, imagem_suave, bordas]

plt.figure(figsize=(10, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(imagens[i], cmap='gray')
    plt.title(titulos[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
