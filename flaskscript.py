from flask import Flask, jsonify, request
app = Flask(__name__)

import numpy as np
from keras.preprocessing import image
from io import BytesIO
from keras.models import model_from_json
from urllib.request import urlopen


def loadImage(URL):
    with urlopen(URL) as url:
        img = image.load_img(BytesIO(url.read()), target_size=(64, 64))

    return image.img_to_array(img)

@app.route('/', methods = ['GET'])
def check():
    return jsonify({'message':'It works!'})

@app.route('/predict', methods = ['POST'])
def predict():
    json_file = open("food_model.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("food_model.h5")
    
    test=loadImage(request.json["url"])
    test=np.expand_dims(test, axis=0)
    
    result=loaded_model.predict(test)
    if (result[0][0]==1):
        outcome='Biryani'
    elif (result[0][1]==1):
        outcome='Burger'
    elif( result[0][2]==1):
        outcome='Chicken Wings'
    elif (result[0][3]==1):
        outcome='Pizza'
    else:
        outcome='Unable Determine'
    
    return jsonify({'result':outcome})
    
if __name__=='__main__':
    app.run(debug=True, port=8080)
    
    
