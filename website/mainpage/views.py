from django.http import HttpResponse
from django.template import loader
import subprocess
import os
from urllib.parse import urlparse
from django.views.decorators.csrf import csrf_exempt
from .predict import predict_single_video
from .models.two_stream import two_stream_model

root_dir = '/Users/cjc/cv'
ts_model = None


def load_model():
    global ts_model
    if ts_model is None:
        spatial_weights = os.path.join(root_dir, 'ActionRecognition/models/finetuned_resnet_RGB_65.h5')
        temporal_weights = os.path.join(root_dir, 'ActionRecognition/models/temporal_cnn_42.h5')
        ts_model = two_stream_model(spatial_weights, temporal_weights)
    return ts_model


@csrf_exempt
def index(request):
    err_msg = ''
    fname = None
    predicted_class = None
    top_number = 5

    try:
        if request.method == 'POST':
            if 'file' not in request.FILES:
                raise FileExistsError('Please choose your video')
            uploaded_file = request.FILES['file']
            fname = uploaded_file.name
            # Write content of the file chunk by chunk in a local file (destination)
            with open('static/' + fname, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            print('write', fname)
            name, ext = os.path.splitext(fname)
            file_dir = os.path.join('static', name)
            if ext != '.mp4' and not os.path.exists(file_dir+'.mp4'):
                proc = subprocess.run(['ffmpeg', '-i', file_dir+ext, '-ac', '2', '-b:v', '2000k', '-c:a', 'aac', '-c:v',
                                       'libx264', '-b:a', '160k', '-vprofile', 'high', '-bf', '0', '-strict',
                                       'experimental', '-f', 'mp4', file_dir+'.mp4'])
                if proc.returncode != 0:
                    raise ChildProcessError('Fail to convert the video format')
                fname = name + '.mp4'
                print('convert video to mp4 format')

        elif request.method == "GET":
            if 'video_link' in request.GET:
                if not os.path.exists('static'):
                    os.mkdir('static')
                video_link = request.GET['video_link']
                if "%20" in video_link:
                    raise FileExistsError('video link should not contain %20')
                fname = urlparse(video_link)
                fname = os.path.basename(fname.path)
                fpath = os.path.join('static', fname)
                if not os.path.exists(fpath):
                    process = subprocess.Popen(["wget", "-P", "static", video_link])
                    process.wait()
            elif 'video' in request.GET:
                fname = request.GET['video']

            if 'predict' in request.GET:
                top_number = int(request.GET['top_number'])
                print("views:", top_number)
                model = load_model()
                predicted_class = predict_single_video(model, os.path.join('static', fname), top_num=top_number)
                print(predicted_class)
    except FileExistsError as err:
        err_msg = str(err)
        print(err_msg)
    finally:
        template = loader.get_template('index.html')
        context = {
            'predicted_class': predicted_class,
            'video': fname,
            'err_msg': err_msg,
            'top_number': top_number,
        }
        print(context)

    return HttpResponse(template.render(context, request))
