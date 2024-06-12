import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

img_pth = "id123_v20.JPG"

image = preprocess(Image.open(img_pth)).unsqueeze(0)
sentences = ["Red box is cat, Blue box is dog", "Red box is dog, Blue box is cat", "Red box is cat", "Blue box is dog", "cat", "dog"]
text = tokenizer(sentences)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(img_pth)
print(sentences)
print(text_probs)  # prints: [[1., 0., 0.]]

img_pth = "id123_v17.JPG"

image = preprocess(Image.open(img_pth)).unsqueeze(0)
sentences = ["ID 001 denotes cat, ID 002 denotes dog", "ID 001 denotes dog, ID 002 denotes cat", "ID 001 denotes cat", "ID 002 denotes dog", "cat", "dog"]
#sentences = ["ID 001 denotes cat, ID 002 denotes dog", "ID 001 denotes dog, ID 002 denotes cat", "ID 001 denotes cat", "ID 002 denotes dog", "cat", "dog"]
#sentences = ["ID 123 denotes cat, ID 456 denotes dog", "ID 123 denotes dog, ID 456 denotes cat", "ID 123 denotes cat", "ID 456 denotes dog", "cat", "dog"]
#sentences = ["ID 123: cat, ID 456: dog", "ID 123: dog, ID 456: cat", "ID 123: cat", "ID 456: dog", "cat", "dog"]

#sentences = ["ID 123 is cat, ID 456 is dog", "ID 123 is dog, ID 456 is cat", "ID 123 is cat", "ID 456 is dog", "cat", "dog"]
#sentences = ["ID-123 is cat, ID-456 is dog", "ID-123 is dog, ID-456 is cat", "ID-123 is cat", "ID-456 is dog", "cat", "dog"]
text = tokenizer(sentences)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(img_pth)
print(sentences)
print(text_probs)  # prints: [[1., 0., 0.]]

img_pth = "id123_v11.JPG"
image = preprocess(Image.open(img_pth)).unsqueeze(0)
sentences = ["red mask cat, blue mask dog", "blue mask cat, red mask dog", "red mask cat", "blue mask dog", "cat", "dog"]
text = tokenizer(sentences)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(img_pth)
print(sentences)
print(text_probs)  # prints: [[1., 0., 0.]]

img_pth = "id123_v10.JPG"
image = preprocess(Image.open(img_pth)).unsqueeze(0)
sentences = ["red circle cat, blue circle dog", "blue circle cat, red circle dog", "red circle cat", "blue circle dog", "cat", "dog"]
text = tokenizer(sentences)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(img_pth)
print(sentences)
print(text_probs)  # prints: [[1., 0., 0.]]
