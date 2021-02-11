from flask import Flask
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)
model = tf.keras.models.load_model("TENSORFLOW MODEL PATH")

test=image.load_img('IMAGE TO TEST PATH',target_size = (64, 64))
test=image.img_to_array(test)
test=np.expand_dims(test, axis=0)

result=model.predict(test)
training_set.class_indices
# print (result)

if (result[0][0]==1):
        outcome='Biryani'
elif (result[0][1]==1):
    outcome='Burger'
elif ( result[0][2]==1):
    outcome='Chicken Wings'
elif (result[0][3]==1):
    outcome='Pizza'
else:
    outcome='Unable Determine'
    
print('result:', outcome)

@app.route('/')
def home():
    return(outcome)

if "__main__" == __name__:
    app.run(debug=True)

# Open browser ang go to localhost:5000 and see result