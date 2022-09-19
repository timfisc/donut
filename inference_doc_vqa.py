from donut import DonutModel
from PIL import Image
import torch


model = DonutModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-docvqa")
question = "What is the total?"
task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
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
