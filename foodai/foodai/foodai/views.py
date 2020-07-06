from django.shortcuts import render
from keras.models import model_from_json
import sys, os, operator

json_file = open("static/food_model.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("static/food_model.h5")



def home(request):
    if request.method == 'POST':
        uploaded_img = request.FILES=['image']
        print (uploaded_img.name)
    return render(request, 'home.html', {})

def login(request):
    return render(request, 'login.html')

def registration(request):
    return render(request, 'registration.html')

def searchresults(request):
    return render(request, 'searchresults.html')