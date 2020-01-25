# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.template import loader, Context
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import home.api as api
import traceback
from io import BytesIO
from PIL import Image
from base64 import b64encode, b64decode

INDEX_FILE = "index_en.html"
generator = api.ImageGenerationAPI("config.json")
model_name = list(generator.models_config.keys())[0]
imsize = generator.models_config[model_name]["output_size"]
base_dic = {
    "imsize" : imsize,
    "canvas_box" : imsize * 2 + 50}

# np.array
def image2bytes(image):
    buffered = BytesIO()
    Image.fromarray(image).save(buffered, format="PNG")
    return b64encode(buffered.getvalue()).decode('utf-8')


def response(image, label, latent):
    imageString = image2bytes(image)
    segString = image2bytes(label)
    latent = b64encode(latent).decode('utf-8')
    json = '{"ok":"true","img":"data:image/png;base64,%s","seg":"data:image/png;base64,%s","latent":"%s"}' % (imageString, segString, latent)
    return HttpResponse(json)


def index(request):
    return render(request, INDEX_FILE, base_dic)


@csrf_exempt
def debug_mask_image(request):
    form_data = request.POST
    if request.method == 'POST' and 'sketch' in form_data and 'model' in form_data:
        try:
            model = form_data['model']
            if not generator.has_model(model):
                return HttpResponse('{}')

            imageData = b64decode(form_data['sketch'].split(',')[1])
            latent = b64decode(form_data['latent'])

            image = Image.open(BytesIO(imageData))
            # TODO: hard coded for stylegan
            sketch, mask = api.stroke2array(image)
            #print("=> edit")
            gen, latent = generator.debug_mask_image(model, mask, latent)
            #print("=> edit done")

            return response(gen, latent)
        except Exception:
            print("!> Exception:")
            traceback.print_exc()
            return HttpResponse('{}')
    return HttpResponse('{}')


@csrf_exempt
def generate_new_image(request):
    form_data = request.POST
    # print(form_data)
    if request.method == 'POST' and 'model' in form_data:
        try:
            model = form_data['model']
            if not generator.has_model(model):
                print("=> No model name %s" % model)
                return HttpResponse('{}')

            image, label, latent = generator.generate_new_image(model)
            return response(image, label, latent)
        except Exception:
            print("!> Exception:")
            traceback.print_exc()
            return HttpResponse('{}')
    return HttpResponse('{}')
