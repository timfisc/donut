from donut import DonutModel
from PIL import Image
import torch


model = DonutModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2")
task_prompt = "<s_cord-v2>"
if torch.cuda.is_available():
    model.half()
    device = torch.device("cuda")
    model.to(device)
else:
    model.encoder.to(torch.bfloat16)
model.eval()

img_path = "dataset/image.jpg"
image = Image.open(img_path).convert("RGB")
output = model.inference(image=image, prompt=task_prompt, return_attentions=False)
print(output)
