import torch 
from torchvision.transforms import transforms
from PIL import Image 
from train import SimpleCNN
model = SimpleCNN()
model.load_state_dict(torch.load('model/simple_cnn.pth'))
model.eval()
transform = transforms.Compose([
     transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor() 
])

def predict(image_path):
    photo = Image.open(image_path).convert('RGB')
    photo = transform(photo)
    photo = photo.unsqueeze(0)
    
    with torch.no_grad():
        output = model(photo)
        _,predicted = torch.max(output,1)
    classes = ["cat" , "dog"]
    return classes[predicted.item()]
if __name__ == "__main__":
    image_path = 'test_image.jpg'  # Provide the path to your test image
    result = predict(image_path)
    print(f'The image is classified as: {result}')
    
    

