import torch
from diffix3d.pipeline_difix import DifixPipeline
from torchvision import transforms

class DiffixUtil:
    """ Utility class for Difix operations.
    """
    def __init__(self):
        print("Initializing DifixUtil...")
        self._pipe = DifixPipeline.from_pretrained("/mnt/public/jason/huggingface/hub/models--nvidia--difix/snapshots/610dcfc6a33c88702e6c4af49336aa55606a7096/")
        self._pipe.to("cuda")
        self._to_tensor = transforms.ToTensor()

    def process_images(self, input_images, ref_image, prompt="remove degradation", 
                       num_inference_steps=1, timesteps=[199], guidance_scale=0.0):
        """ Process input and reference images with the Difix pipeline. """
        image_list = []
        for image_index in range(input_images.shape[1]):
            output_image =  self._pipe(prompt, image=input_images[0, image_index], ref_image=ref_image[0][0],
                                       height =input_images[0, image_index].shape[1],
                                       width = input_images[0, image_index].shape[2], 
                                       num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
            image_tensor = self._to_tensor(output_image).cuda()
            image_list.append(image_tensor)
        output_images = torch.stack(image_list, dim=0).unsqueeze(0)
        return output_images


