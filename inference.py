import torch
from diffusers import StableDiffusionPipeline

model_path  = "C:/Users/anushreeasthana/Desktop/diffusers/examples/text_to_image/sd-model-finetuned"

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="enter your prompt here").images[0]
image.save("cafe3.png")
