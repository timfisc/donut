from pprint import pprint as pp
import json
import glob
import os

import pypdfium2 as pdfium
from PIL import Image
from tqdm import tqdm
import torch

from donut import DonutModel, DonutModelCustom


model_path = "/home/ss3/donut/result/donut_finetuned_models/experiment_7k_ver1_lr2e5_60ep"
model_path = "/home/ss3/donut/result/donut_finetuned_models/experiment_2k_ver2_lr2e5_500epoch"
model_path = "/home/ss3/donut/result/donut_finetuned_models/experiment_7k_ver1_lr2e5_500epoch"
custom_model = True

if custom_model:
    mdoel = DonutModelCustom.from_pretrained(model_path)
else:
    model = DonutModel.from_pretrained(model_path)

if torch.cuda.is_available():
    model.half()
    device = torch.device("cuda")
    model.to(device)
else:
    model.encoder.to(torch.bfloat16)
model.eval()

img_ext = {".jpg", ".jpeg", ".JPEG", ".png", ".PNG"}
pdf_ext = {".pdf", ".PDF"}
valid_ext = img_ext | pdf_ext

src_dir = "dataset/scanned_invoice"
img_paths = glob.glob(os.path.join(src_dir, "**/*.*"), recursive=True)
dest_dir = os.path.join(src_dir, f"json_{model_path.rstrip('/').split('/')[-1]}")
os.makedirs(dest_dir, exist_ok=True)

img_paths = [path for path in img_paths if os.path.splitext(path)[1] in valid_ext]

for img_path in tqdm(img_paths):
    img_path_wout_ext, ext = os.path.splitext(img_path)
    if ext in img_ext:
        image = Image.open(img_path).convert("RGB")
    elif ext in pdf_ext:
        with pdfium.PdfDocument(img_path) as pdf:
            image = [img for img in pdf.render_topil(scale=2)][0]
    else:
        print(f"{img_path} not recognized as an image or pdf file")
        continue
    output = model.inference(image=image, prompt="<s_iitcdip>")
    print(img_path)
    pp(output)

    img_name = img_path_wout_ext.split('/')[-1] + ".json"
    dest_json_path = os.path.join(dest_dir, img_name)
    with open(dest_json_path, 'w') as f:
        f.write(json.dumps(output))
