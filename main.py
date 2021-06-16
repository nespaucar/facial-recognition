from flask import Flask, render_template, Response # Flask para interpretar código python en la web mediante un puerto
from camera import VideoCamera # Invocamos a camera.py para abrir la cámara en la web

# Inicializamos Flask

app = Flask(__name__)

# Llamamos a la cámara del sistema

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Creamos las rutas para ser leídas por la web

@app.route('/') # Ruta principal
def index():
    return render_template('index.html')

@app.route('/modelo') # Ruta para ver el modelo en un gráfico
def modelo():
    return render_template('modelo.html')

@app.route('/video_feed') # Ruta para llamar a la cámara
def video_feed():
    return Response(gen(VideoCamera()), mimetype = 'multipart/x-mixed-replace; boundary=frame')

# Definimos puerto para desplegar sistema en la web y le definimos como en desarrollo True para detección de cambios

if __name__ == '__main__':
    app.run(host='localhost', debug = True)
