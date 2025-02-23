from flask import Flask, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
import base64

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

IMG_SIZE = (224, 224)

try:
    resnet_model = tf.keras.models.load_model('tom_jerry_resnet_model.keras')
    densenet_model = tf.keras.models.load_model('tom_jerry_densenet_model.keras')
except Exception as e:
    print("Error loading models:", e)
    raise e

class_names = ['jerry', 'tom', 'tom_jerry_0', 'tom_jerry_1']

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def map_prediction(pred):
    """Map model prediction to a friendly display text."""
    if pred == 'jerry':
        return "Jerry"
    elif pred == 'tom':
        return "Tom"
    elif pred == 'tom_jerry_0':
        return "Both not found"
    elif pred == 'tom_jerry_1':
        return "Found both Tom & Jerry"
    else:
        return pred

@app.route('/')
def index():
    html = '''
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Image Prediction</title>
      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
      <style>
        body { background-color: #f8f9fa; }
        .container { margin-top: 50px; max-width: 600px; }
        #previewImg { margin-top: 20px; max-width: 100%; display: none; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="text-center">
          <h1>4JaturatepPordee</h1>
          <h2>Tom &amp; Jerry Image Classification</h2>
        </div>
        <h3 class="text-left" style="margin-top:30px;">Upload an Image for Prediction</h3>
        <form method="POST" action="/predict" enctype="multipart/form-data">
          <div class="form-group">
            <label for="file">Choose an image file</label>
            <input type="file" class="form-control-file" name="file" id="file" required onchange="previewFile()">
          </div>
          <img id="previewImg" src="#" alt="Image Preview">
          <button type="submit" class="btn btn-primary btn-block" style="margin-top:20px;">Upload &amp; Predict</button>
        </form>
      </div>
      
      <script>
      function previewFile() {
          const file = document.getElementById('file').files[0];
          const preview = document.getElementById('previewImg');
          const reader = new FileReader();
          reader.onloadend = function() {
              preview.src = reader.result;
              preview.style.display = "block";
          }
          if (file) {
              reader.readAsDataURL(file);
          } else {
              preview.src = "";
              preview.style.display = "none";
          }
      }
      </script>
    </body>
    </html>
    '''
    return html

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part in the request."
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected."
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        with open(filepath, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        ext = filename.rsplit('.', 1)[1].lower()
        if ext == "png":
            mime = "image/png"
        elif ext == "gif":
            mime = "image/gif"
        else:
            mime = "image/jpeg"
        
        new_image = tf.keras.preprocessing.image.load_img(filepath, target_size=IMG_SIZE)
        new_image = tf.keras.preprocessing.image.img_to_array(new_image)
        new_image = tf.expand_dims(new_image, 0)
        
        # Prediction using ResNet50
        resnet_predictions = resnet_model.predict(new_image)
        resnet_index = tf.argmax(resnet_predictions[0]).numpy()
        if resnet_index < len(class_names):
            resnet_pred = class_names[resnet_index]
        else:
            resnet_pred = "Unknown"
        resnet_display = map_prediction(resnet_pred)
        
        # Prediction using DenseNet121
        densenet_predictions = densenet_model.predict(new_image)
        densenet_index = tf.argmax(densenet_predictions[0]).numpy()
        if densenet_index < len(class_names):
            densenet_pred = class_names[densenet_index]
        else:
            densenet_pred = "Unknown"
        densenet_display = map_prediction(densenet_pred)
        
        os.remove(filepath)
        
        html = f'''
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title>Prediction Results</title>
          <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
          <style>
            body {{ background-color: #f8f9fa; }}
            .container {{ margin-top: 50px; max-width: 600px; }}
            .result-card {{ margin-top: 20px; }}
            .img-preview {{ max-width: 100%; margin-top: 20px; }}
          </style>
        </head>
        <body>
          <div class="container">
            <h1 class="text-center">Prediction Results</h1>
            <div class="text-center">
              <img src="data:{mime};base64,{encoded_image}" alt="Uploaded Image" class="img-preview">
            </div>
            <div class="card result-card">
              <div class="card-body">
                <h5 class="card-title">ResNet50 Prediction</h5>
                <p class="card-text">{resnet_display}</p>
              </div>
            </div>
            <div class="card result-card">
              <div class="card-body">
                <h5 class="card-title">DenseNet121 Prediction</h5>
                <p class="card-text">{densenet_display}</p>
              </div>
            </div>
            <div class="text-center" style="margin-top:20px;">
              <a href="/" class="btn btn-secondary">Upload Another Image</a>
            </div>
          </div>
        </body>
        </html>
        '''
        return html
    else:
        return "File type not allowed."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3006)