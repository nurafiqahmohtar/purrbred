from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np

#model = Sequential()

#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,3)))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Flatten())
#model.add(Dense(512, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))
#model.add(Dense(2, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.load_weights('static/model.h5')

#medium CNN architecture
model = Sequential()
model.add(Conv2D(100, (5,5),input_shape=(224,224,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4, 4), strides=4))
model.add(Conv2D(50, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(8, 8), strides=8))
model.add(Flatten())
model.add(Dropout( 0.5))
model.add(Dense(50))
model.add(Activation("relu"))
model.add(Dense(25))
model.add(Activation("relu"))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=["accuracy"])
model.load_weights('static/model_weights.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224,224,3)
    prediction = model.predict(img_arr)

    a = round(prediction[0,0], 2)
    b = round(prediction[0,1], 2)
    c = round(prediction[0,2], 2)
    d = round(prediction[0,3], 2)
    e = round(prediction[0,4], 2)
    preds = np.array([a,b,c,d,e])
    COUNT += 1
    return render_template('prediction.html', data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



