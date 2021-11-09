import torch
from torchvision import models
import numpy as np

print(torch.cuda.is_available())

image = np.random.random(size=[2, 3, 224, 224])
image.dtype = 'float32'

image_tensor = torch.from_numpy(image).cuda()

model = models.resnet50(pretrained=True)
model = model.cuda()

out = model(image_tensor)
print(out)

# IPython.embed()
