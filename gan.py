"""
GAN sencilla sobre el dataset Quick, Draw! de Google (categoría: cookie).

Flujo:
  1. Lee full_raw_cookie.ndjson (trazos crudos) y convierte cada dibujo en un bitmap 28x28.
  2. Crea la red GENERADORA (G) y la red DISCRIMINADORA (D).
  3. Las entrena una contra la otra con la pérdida BCE (Binary Cross Entropy).
  4. Exporta parámetros, pérdidas, valores y dibujos generados a web/ para el frontend.

Uso:  python gan.py
      python gan.py --file full_raw_otra.ndjson --epochs 40
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Hiperparámetros (se pueden cambiar por línea de comandos)
# ----------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--file", default="full_raw_cookie.ndjson")  # archivo crudo de Quick Draw
p.add_argument("--samples", type=int, default=20000)  # dibujos usados para entrenar
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--z_dim", type=int, default=64)   # tamaño del vector de ruido z
p.add_argument("--lr", type=float, default=2e-4)  # learning rate de ambos optimizadores
args = p.parse_args()

torch.manual_seed(0)
IMG = 28 * 28  # cada dibujo es 28x28 = 784 píxeles
os.makedirs("data", exist_ok=True)
os.makedirs("web/img", exist_ok=True)

# ----------------------------------------------------------------------------
# 1. DATOS: trazos crudos (ndjson) -> bitmaps 28x28
#    Cada línea del archivo es un JSON con:
#      word, countrycode, timestamp, recognized, key_id,
#      drawing = [ [[x0,x1,...],[y0,y1,...],[t0,t1,...]], ...un elemento por trazo ]
#    Las coordenadas están en píxeles de la pantalla del usuario, sin normalizar.
# ----------------------------------------------------------------------------
def rasterize(strokes, size=28, canvas=256, width=14):
    """Escala el dibujo a un lienzo de 256x256, dibuja los trazos y lo reduce a 28x28."""
    xs = np.concatenate([s[0] for s in strokes]); ys = np.concatenate([s[1] for s in strokes])
    scale = (canvas - 2 * width) / max(xs.max() - xs.min(), ys.max() - ys.min(), 1)
    # centrar el dibujo en el lienzo conservando su proporción
    ox = (canvas - (xs.max() - xs.min()) * scale) / 2 - xs.min() * scale
    oy = (canvas - (ys.max() - ys.min()) * scale) / 2 - ys.min() * scale
    img = Image.new("L", (canvas, canvas), 0)
    d = ImageDraw.Draw(img)
    for s in strokes:
        pts = [(x * scale + ox, y * scale + oy) for x, y in zip(s[0], s[1])]
        d.line(pts if len(pts) > 1 else pts * 2, fill=255, width=width, joint="curve")
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.float32)


records = []
with open(args.file, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["recognized"]:  # solo dibujos que el juego reconoció como la categoría
            records.append(r)
            if len(records) == args.samples:
                break
category = records[0]["word"]  # categoría usada (viene dentro del propio archivo)
x = np.stack([rasterize(r["drawing"]) for r in records])
assert x.shape == (len(records), 28, 28) and x.max() > 0
x = x.reshape(len(x), IMG) / 127.5 - 1.0  # píxeles de [0,255] a [-1,1], igual que la salida Tanh del generador
data = torch.utils.data.DataLoader(torch.from_numpy(x), batch_size=args.batch, shuffle=True, drop_last=True)


# ----------------------------------------------------------------------------
# 2. RED NEURONAL GENERADORA (G)
#    Entrada: vector de ruido z (z_dim números aleatorios)
#    Salida : un dibujo falso de 784 píxeles en [-1,1]
# ----------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),   # capa 1: z -> 256
            nn.Linear(256, 512), nn.LeakyReLU(0.2),     # capa 2: 256 -> 512
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),    # capa 3: 512 -> 1024
            nn.Linear(1024, IMG), nn.Tanh(),            # capa de salida: 1024 -> 784 píxeles
        )

    def forward(self, z):
        return self.net(z)


# ----------------------------------------------------------------------------
# 3. RED NEURONAL DISCRIMINADORA (D)
#    Entrada: un dibujo de 784 píxeles (real o generado)
#    Salida : probabilidad de que el dibujo sea REAL (Sigmoid -> [0,1])
# ----------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),  # capa 1: 784 -> 512
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),  # capa 2: 512 -> 256
            nn.Linear(256, 1), nn.Sigmoid(),                          # salida: 256 -> 1 probabilidad
        )

    def forward(self, img):
        return self.net(img)


# >>> AQUÍ SE CREAN LAS DOS REDES <<<
G = Generator(args.z_dim)   # red generadora
D = Discriminator()         # red discriminadora

# Auto-chequeo de formas: G produce 784 píxeles, D produce 1 probabilidad por imagen
assert G(torch.randn(2, args.z_dim)).shape == (2, IMG)
assert D(torch.randn(2, IMG)).shape == (2, 1)

# ----------------------------------------------------------------------------
# 4. FUNCIÓN DE PÉRDIDA Y OPTIMIZADORES
#    BCE(p, y) = -[ y*log(p) + (1-y)*log(1-p) ]
#    Pérdida D = BCE(D(x), 1) + BCE(D(G(z)), 0)  -> aprender a distinguir real/falso
#    Pérdida G = BCE(D(G(z)), 1)                 -> engañar a D (variante "non-saturating")
# ----------------------------------------------------------------------------
bce = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))


def layers_info(model):
    """Parámetros y valores (media/desv./mín./máx. de los pesos) de cada capa Linear."""
    out = []
    for name, m in model.net.named_children():
        if isinstance(m, nn.Linear):
            w = m.weight.detach()
            out.append({"layer": f"Linear {m.in_features} -> {m.out_features}",
                        "weights": w.numel(), "bias": m.bias.numel(),
                        "mean": w.mean().item(), "std": w.std().item(),
                        "min": w.min().item(), "max": w.max().item()})
    return out


def save_grid(imgs, file, n=8):
    """Guarda n x n dibujos como una sola imagen PNG."""
    imgs = ((imgs[: n * n].reshape(n, n, 28, 28) + 1) / 2).clamp(0, 1).numpy()
    grid = imgs.transpose(0, 2, 1, 3).reshape(n * 28, n * 28)
    plt.imsave(file, 1 - grid, cmap="gray")  # 1 - grid: trazo negro sobre fondo blanco


save_grid(torch.from_numpy(x[:64]), "web/img/real.png")
fixed_z = torch.randn(64, args.z_dim)  # mismo ruido en cada época para ver la evolución
history = []

# ----------------------------------------------------------------------------
# 5. ENTRENAMIENTO ADVERSARIAL
# ----------------------------------------------------------------------------
for epoch in range(1, args.epochs + 1):
    sums = np.zeros(4)
    for real in data:
        b = real.size(0)
        ones, zeros = torch.ones(b, 1), torch.zeros(b, 1)

        # --- Paso del DISCRIMINADOR: reales -> 1, falsos -> 0 ---
        fake = G(torch.randn(b, args.z_dim))
        d_real, d_fake = D(real), D(fake.detach())  # detach: no actualizar G aquí
        loss_D = bce(d_real, ones) + bce(d_fake, zeros)
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # --- Paso del GENERADOR: quiere que D diga 1 a sus falsos ---
        loss_G = bce(D(fake), ones)
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        sums += [loss_D.item(), loss_G.item(), d_real.mean().item(), d_fake.mean().item()]

    loss_d, loss_g, dx, dgz = sums / len(data)
    history.append({"epoch": epoch, "loss_D": loss_d, "loss_G": loss_g, "D_x": dx, "D_G_z": dgz})
    print(f"época {epoch:3d}  loss_D={loss_d:.3f}  loss_G={loss_g:.3f}  D(x)={dx:.3f}  D(G(z))={dgz:.3f}")
    G.eval()
    with torch.no_grad():
        save_grid(G(fixed_z), f"web/img/epoch_{epoch:03d}.png")
    G.train()

# ----------------------------------------------------------------------------
# 6. EXPORTAR RESULTADOS AL FRONTEND (data.js se carga con <script>, sin servidor)
# ----------------------------------------------------------------------------
count = lambda m: sum(t.numel() for t in m.parameters())
result = {
    "category": category, "samples": len(x), "hyper": vars(args),
    # 6 dibujos crudos (solo x,y de cada trazo) para mostrar el antes/después en la web
    "raw": [[[s[0], s[1]] for s in r["drawing"]] for r in records[:6]],
    "raw_meta": {k: v for k, v in records[0].items() if k != "drawing"},
    "generator": {"total": count(G), "layers": layers_info(G), "text": str(G)},
    "discriminator": {"total": count(D), "layers": layers_info(D), "text": str(D)},
    "history": history,
}
with open("web/data.js", "w", encoding="utf-8") as f:
    f.write("window.GAN = " + json.dumps(result, indent=1) + ";")
torch.save({"G": G.state_dict(), "D": D.state_dict()}, "data/gan.pt")
print("Listo: abre web/index.html")
