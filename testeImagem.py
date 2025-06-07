import cv2
import matplotlib.pyplot as plt

# Lê a imagem (lembre-se de substituir pelo caminho da sua imagem)
imagem = cv2.imread('wizarding-world-legacy-link-reward-wallpaper-0eaee25c51d148b5ada70c8944b7e199.jpg')

# OpenCV lê as imagens no formato BGR, então vamos converter para RGB
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Exibe a imagem
plt.imshow(imagem_rgb)
plt.title("Imagem Original")
plt.axis('off')
plt.show()

# Exibe os dados da imagem
print("Dimensões:", imagem_rgb.shape)  # (altura, largura, 3)
print("Pixel no canto superior esquerdo (R, G, B):", imagem_rgb[0, 0])
