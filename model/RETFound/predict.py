import torch
from PIL import Image
import torchvision.transforms as transforms
import models_vit  # from RETFound repo

# -------- CONFIG --------
MODEL_PATH = "output_dir/checkpoint-best.pth"
IMAGE_PATH = "test.png"   # change this
NUM_CLASSES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD MODEL --------
model = models_vit.RETFound_mae(
    img_size=256,
    num_classes=NUM_CLASSES,
    drop_path_rate=0.1,
    global_pool=True
)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"])

model.to(device)
model.eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------- LOAD IMAGE --------
image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# -------- PREDICT --------
with torch.no_grad():
    output = model(image)
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(probs, dim=1).item()

# -------- LABELS --------
classes = ["normal", "glaucoma"]

print("\nPrediction:", classes[pred])