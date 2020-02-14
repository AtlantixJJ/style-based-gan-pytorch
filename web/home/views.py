# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.template import loader, Context
from django.http import HttpResponse
#from django.views.decorators.csrf import csrf_exempt
import home.api as api
import traceback
from io import BytesIO
from PIL import Image
from base64 import b64encode, b64decode

INDEX_FILE = "index_en.html"
editor = api.ImageGenerationAPI("config.json")
model_name = list(editor.models_config.keys())[0]
imsize = editor.models_config[model_name]["output_size"]
base_dic = {
    "imsize" : imsize,
    "canvas_box" : imsize * 2 + 50}

# np.array
def image2bytes(image):
    buffered = BytesIO()
    Image.fromarray(image).save(buffered, format="PNG")
    return b64encode(buffered.getvalue()).decode('utf-8')


def response(image, label):
    imageString = image2bytes(image)
    segString = image2bytes(label)
    #latent = b64encode(latent).decode('utf-8')
    #noise = b64encode(noise).decode('utf-8')
    #json = '{"ok":"true","img":"data:image/png;base64,%s","label":"data:image/png;base64,%s","latent":"%s","noise":"%s"}' % (imageString, segString, latent, noise)
    json = '{"ok":"true","img":"data:image/png;base64,%s","label":"data:image/png;base64,%s"}' % (imageString, segString)
    return HttpResponse(json)


def index(request):
    return render(request, INDEX_FILE, base_dic)


def generate_image_given_stroke(request):
    form_data = request.POST
    sess = request.session
    if request.method == 'POST' and 'image_stroke' in form_data:
        try:
            model = form_data['model']
            if not editor.has_model(model):
                print(f"!> Model not exist {model}")
                return HttpResponse('{}')

            imageStrokeData = b64decode(form_data['image_stroke'].split(',')[1])
            labelStrokeData = b64decode(form_data['label_stroke'].split(',')[1])
            #latent = b64decode(form_data['latent'])
            #noise = b64decode(form_data['noise'])
            latent = sess["latent"]
            noise = sess["noise"]

            imageStroke = Image.open(BytesIO(imageStrokeData))
            labelStroke = Image.open(BytesIO(labelStrokeData))
            # TODO: hard coded for stylegan
            imageStroke, imageMask = api.stroke2array(imageStroke)
            labelStroke, labelMask = api.stroke2array(labelStroke)

            image, label, latent, noise = editor.generate_image_given_stroke(
                model, latent, noise,
                imageStroke, imageMask,
                labelStroke, labelMask)
            sess["latent"] = latent
            sess["noise"] = noise
            return response(image, label)
        except Exception:
            print("!> Exception:")
            traceback.print_exc()
            return HttpResponse('{}')
    print(f"!> Invalid request: {str(form_data.keys())}")
    return HttpResponse('{}')


def generate_new_image(request):
    form_data = request.POST
    sess = request.session
    if request.method == 'POST' and 'model' in form_data:
        try:
            model = form_data['model']
            if not editor.has_model(model):
                print("=> No model name %s" % model)
                return HttpResponse('{}')

            image, label, latent, noise = editor.generate_new_image(model)
            sess["latent"] = latent
            sess["noise"] = noise
            return response(image, label)
        except Exception:
            print("!> Exception:")
            traceback.print_exc()
            return HttpResponse('{}')
    return HttpResponse('{}')
