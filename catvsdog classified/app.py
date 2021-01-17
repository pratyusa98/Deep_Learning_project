from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename
import os, sys, glob, re
import tensorflow_hub as hub




app = Flask(__name__)


model = load_model(('catvsdog.h5'),custom_objects={'KerasLayer':hub.KerasLayer})



def model_predict(image_path,model):
    print("Predicted")
    image = load_img(image_path,target_size=(224,224))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = model.predict(image)

    if result[0]<=0.5:
        result = "The image classified is cat"
        return result
    else:
        result = "The image classified is dog"
        return result



IMG_FOLDER = os.path.join('static', 'upload')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
	if request.method == 'POST':
		file = request.files['image'] # fetch input
		filename = file.filename 

		file_path = os.path.join('static/upload', filename)
		file.save(file_path)

		pred = model_predict(file_path,model)

		full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)


		return render_template('index.html',predict = pred,user_image = full_filename)




		


if __name__ == '__main__':
    app.run(debug=True)
    