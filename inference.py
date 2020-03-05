import torch
from PIL import Image
from torchvision import transforms,models

filename="/media/data/dataset/val/0501.jpg"
model_file_path="/media/pytorch_env/epoch_80.pth"
#original GPU,CPU inference
# read model
model =torch.load(model_file_path, map_location=lambda storage, loc: storage)
#model = models.mobilenet_v2(pretrained=True) # if original Official

model.eval()


input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available,now cpu
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

print(output)
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))
_, preds = torch.max(output, 1)
print(preds)
