from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("/root/.cache/huggingface/hub/models--nvidia--difix/snapshots/610dcfc6a33c88702e6c4af49336aa55606a7096/")
pipe.to("cuda")

input_image = load_image("assets/example_input.png")
ref_image = load_image("assets/example_ref.png")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
output_image.save("example_output.png")