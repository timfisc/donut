from donut import DonutModel, DonutModelCustom
from PIL import Image
import torch

custom_model = True

if custom_model:
    mdoel = DonutModelCustom.from_pretrained(
        "result/train_sroie/20220816_113929/")
else:
    model = DonutModel.from_pretrained(
        "result/train_sroie/20220816_113929/")

if torch.cuda.is_available():
    model.half()
    device = torch.device("cuda")
    model.to(device)
else:
    model.encoder.to(torch.bfloat16)
model.eval()

img_path = "/home/ss3/donut/dataset/task3-test 347p)/task3-test（347p)/X00016469670.jpg"
img_path = "./dataset/sroie-donut/test/X51005441401.jpg"
img_path = "dataset/3kfc.jpg"
image = Image.open(img_path).convert("RGB")
output = model.inference(image=image, prompt="<s_sroie-donut>")
print(output)
