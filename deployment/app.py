from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as T
import io

app = Flask(__name__)

model_path = 'model/transfer_exported.pt'
model = torch.jit.load(model_path)
model.eval()

transform = T.Compose([
    T.Resize([256, ]),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    img = Image.open(io.BytesIO(file.read()))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)

    class_name = model.class_names[predicted.item()]
    return f'Predicted: {class_name}'

if __name__ == '__main__':
    app.run(debug=True)
