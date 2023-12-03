import requests
import numpy as np
import os
import PIL
from flask import Flask, app, request, render_template, redirect, url_for
from keras import models
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from keras.applications.inception_v3 import preprocess_input

model = load_model(r"Updated-Xception-diabetic-retinopathy.h5")
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def home():
    return render_template("index.html")

from sqlite3 import connect
from cloudant.client import Cloudant
client = Cloudant.iam("21476c5e-bfa8-40d1-8bad-a86acfa23cd8-bluemix", "pAD3eLwPhCm9JcEKtEriPH3sESRmdHJ8CshemxdvtrRr", connect = True)
my_database = client.create_database('my_database')

@app.route('/register.html')
def register():
  return render_template('register.html')

@app.route('/afterreg', methods=['POST'])
def afterreg():
  x = [x for x in request.form.values()]
  print(x)
  data = {
      '_id' : x[1],
      'name' : x[0],
      'psw' : x[2]
  }
  print(data)

  query = {'_id' : {'Seq' : data['_id']}}

  docs = my_database.get_query_result(query)
  print(docs)

  print(len(docs.all()))

  if((len(docs.all())) == 0):
    url = my_database.create_document(data)
    return render_template('register.html', pred="Register Success")
  else:
    return render_template('register.html', pred="Already a Member")

@app.route('/login.html')
def login():
  return render_template('login.html')

@app.route('/afterlogin', methods=['POST'])
def afterlogin():
  user = request.form['_id']
  passw = request.form['psw']
  print(user, passw)
  
  query = {'_id': {'$eq': user}}

  docs = my_database.get_query_result(query)
  print(docs)

  print(len(docs.all()))

  if(len(docs.all()) == 0):
    return render_template('login.html', pred="User not found")
  else:
    if((user==docs[0][0]['_id'] and passw == docs[0][0]['psw'])):
      return redirect(url_for('prediction'))
    else:
      print('invalid user')

@app.route('/logout')
def logout():
  return render_template('logout.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/result',methods = ["GET","POST"])
def res():
    if request.method == "POST":
        f=request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img = image.load_img(filepath,target_size=(299,299))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)

        index = ['No Diabetic Retinopathy', 'Mild Diabetic Retinopathy', 'Moderate Diabetic Retinopathy', 'Severe Diabetic Retinopathy', 'Proliferative Diabetic Retinopathy']
        result = str(index[ prediction[0]])
        print(result)
        return render_template('prediction.html',prediction=result)
        
if __name__ == "__main__":
    app.run(debug=True)