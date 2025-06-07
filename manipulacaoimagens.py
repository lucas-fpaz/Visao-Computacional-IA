import cv2
import matplotlib.pyplot as plt

# Lê a imagem original
imagem = cv2.imread('wizarding-world-legacy-link-reward-wallpaper-0eaee25c51d148b5ada70c8944b7e199.jpg')
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# 1. Converter para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# 2. Redimensionar a imagem
imagem_menor = cv2.resize(imagem_rgb, (100, 100))

# 3. Cortar (crop): região do centro
altura, largura, _ = imagem_rgb.shape
corte = imagem_rgb[altura//4:3*altura//4, largura//4:3*largura//4]

# 4. Rotacionar a imagem (90 graus)
imagem_rotacionada = cv2.rotate(imagem_rgb, cv2.ROTATE_90_CLOCKWISE)

# 5. Espelhamento horizontal
imagem_espelhada = cv2.flip(imagem_rgb, 1)

# Exibir tudo
titulos = ['Original', 'Cinza', 'Menor', 'Corte', 'Rotacionada', 'Espelhada']
imagens = [imagem_rgb, imagem_cinza, imagem_menor, corte, imagem_rotacionada, imagem_espelhada]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    if len(imagens[i].shape) == 2:  # imagem em cinza
        plt.imshow(imagens[i], cmap='gray')
    else:
        plt.imshow(imagens[i])
    plt.title(titulos[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
