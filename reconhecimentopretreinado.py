# Esse exemplo usa OpenCV + YOLOv3
import cv2
import numpy as np

# Carrega os nomes das classes (ex: person, car, dog...)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Carrega modelo YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Carrega imagem
imagem = cv2.imread('Pinheiros_Avenida-Faria-Lima_Rua-Butanta_Foto-Marcos-Santos_U0Y1378-scaled.jpg')
altura, largura = imagem.shape[:2]

# Pré-processamento
blob = cv2.dnn.blobFromImage(imagem, 1/255, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Pega camadas de saída
camadas_saida = net.getUnconnectedOutLayersNames()
saidas = net.forward(camadas_saida)

# Processa as detecções
for saida in saidas:
    for detecao in saida:
        scores = detecao[5:]
        classe_id = np.argmax(scores)
        confianca = scores[classe_id]

        if confianca > 0.5:
            centro_x = int(detecao[0] * largura)
            centro_y = int(detecao[1] * altura)
            w = int(detecao[2] * largura)
            h = int(detecao[3] * altura)

            x = int(centro_x - w / 2)
            y = int(centro_y - h / 2)

            # Desenha a caixa e o nome da classe
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
            texto = f"{classes[classe_id]} ({int(confianca*100)}%)"
            cv2.putText(imagem, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Exibe imagem
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
import matplotlib.pyplot as plt
plt.imshow(imagem_rgb)
plt.title("Reconhecimento de Objetos com YOLO")
plt.axis('off')
plt.show()
