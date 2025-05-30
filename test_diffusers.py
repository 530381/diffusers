from diffusers import DiffusionPipeline
import torch

# 加载预训练模型
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

# 生成图像
image = pipeline("An image of a squirrel in Picasso style").images[0]

# 保存图像
image.save("generated_image.png")
print("Image generated and saved as generated_image.png")
