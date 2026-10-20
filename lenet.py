# ============================================================
# prever.py — Classificador de imagem única com LeNet5Improved
# ============================================================
import argparse
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ------------------------------------------------------------
# 1. Arquitetura (cópia EXATA do treino)
# ------------------------------------------------------------
class LeNet5Improved(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        return self.fc3(x)


# ------------------------------------------------------------
# 2. Configuração (classes e transformação de inferência)
# ------------------------------------------------------------
CLASSES = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo',
           'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

TRANSFORM_INFER = transforms.Compose([
    transforms.Resize(36),         # menor lado vira 36
    transforms.CenterCrop(32),     # recorta o centro 32x32 (mantém proporção)
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])


# ------------------------------------------------------------
# 3. Carregar modelo
# ------------------------------------------------------------
def carregar_modelo(caminho_pesos: str, device):
    model = LeNet5Improved(num_classes=10).to(device)
    state = torch.load(caminho_pesos, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------------------------------------
# 4. Inferência em uma imagem
# ------------------------------------------------------------
@torch.no_grad()
def prever(model, imagem: Image.Image, device, top_k: int = 3):
    """Recebe uma PIL.Image e retorna lista [(classe, prob), ...]."""
    tensor = TRANSFORM_INFER(imagem.convert('RGB')).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    top = probs.topk(top_k)
    return [(CLASSES[idx.item()], prob.item())
            for prob, idx in zip(top.values, top.indices)]


# ------------------------------------------------------------
# 5. CLI
# ------------------------------------------------------------
def main_gui():
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = carregar_modelo('best_model.pth', device)

    root = tk.Tk()
    root.title('Classificador CIFAR-10 — LeNet5Improved')
    root.geometry('600x500')

    # Área para mostrar a imagem
    lbl_img = tk.Label(root)
    lbl_img.pack(pady=10)

    # Texto do resultado
    lbl_res = tk.Label(root, text='Escolha uma imagem...', font=('Arial', 14))
    lbl_res.pack(pady=10)

    def escolher():
        caminho = filedialog.askopenfilename(
            title='Escolha uma imagem',
            filetypes=[('Imagens', '*.png *.jpg *.jpeg *.bmp *.gif')]
        )
        if not caminho:
            return

        img = Image.open(caminho)
        img_show = img.copy()
        img_show.thumbnail((400, 400))
        foto = ImageTk.PhotoImage(img_show)
        lbl_img.config(image=foto)
        lbl_img.image = foto

        resultados = prever(model, img, device, top_k=3)
        texto = '\n'.join(
            f'{i}. {c:12s} {p*100:6.2f}%'
            for i, (c, p) in enumerate(resultados, 1)
        )
        lbl_res.config(text=f'Predição: {resultados[0][0].upper()}\n\n{texto}')

    tk.Button(root, text='Escolher imagem', command=escolher,
              font=('Arial', 12), padx=20, pady=8).pack(pady=10)

    root.mainloop()


if __name__ == '__main__':
    import sys
    if '--gui' in sys.argv:
        main_gui()
    else:
        main_gui()


if __name__ == '__main__':
    main_gui()