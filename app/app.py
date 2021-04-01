# Importation
import base64
import io
import pickle
import yaml


import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf

from dash.dependencies import Input, Output
from constants import CLASSES
from PIL import Image


#Téléchargement des paramètres de app.yaml
with open('app.yaml') as yaml_app:
    parameters = yaml.safe_load(yaml_app)
    
IMAGE_WIDTH = parameters['parameters']['IMAGE_WIDTH']
IMAGE_HEIGHT = parameters['parameters']['IMAGE_HEIGHT']
load_model_path_DNN = parameters['paths']['load_model_path_DNN']
load_model_path_SVM = parameters['paths']['load_model_path_SVM']

#Téléchargement des modèles
DNN = tf.keras.models.load_model(load_model_path_DNN)
SVM = pickle.load(open(load_model_path_SVM,'rb'))

#Prédiction de la classe de l'image et affichage des probabilités 
def classify_image(image, model, image_box=None):
    """Classify image by model and show probabilities for each traffic signs
    
    Parameters
    ---------
    content: image content
    model: tf/keras classifier
    
    Returns
    -------
    class id returned by model classifier
    list of probabilities for each traffic signs
    """
    
    images_list = []
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box)
                                        # box argument clips image to (x1, y1, x2, y2)
    image = np.array(image)
    
    if model=='SVM':
        image_color = image.flatten()
        # convert images to greyscale
        image =rgb2grey(image)
        # combine color and hog features into a single array
        flat_features = np.hstack(image_color)
        images_list.append(flat_features)
        
        return model.predict(np.array(images_list)), model.predict_proba(np.array(images_list)).tolist()[0]
    
    else:
        images_list.append(image)
        return model.predict_classes(np.array(images_list)), model.predict(np.array(images_list)).tolist()[0]

# Initialisation de l'app

app = dash.Dash('Traffic Signs Recognition') 

colors = {
    'background': '#dfeef9',
    'text': '#203675'
}

pre_style = {
    'whiteSpace': 'pre-wrap',
    'wordBreak': 'break-all',
    'whiteSpace': 'normal'
}



# Layout de l'app
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(children='Prédiction des types de panneaux',
            style={
            'textAlign': 'center',
            'color': colors['text'],
            'verticalAlign': 'middle',
        }), 
     html.Div(children='Prédiction des types de panneaux et affichage des probabilités', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Upload(
        id='bouton-chargement',
        children=html.Div([
            'Cliquer-déposer ou ',
                    html.A('sélectionner une image')
        ]),
        style={
            'width': '25%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '5px',
            'borderStyle': 'double',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '25px',
            'color': colors['text'],
            'marginLeft': 'auto',
            'marginRight': 'auto'
        }
    ),
    html.Div(id='mon-image')
])

#Callback 
@app.callback(Output('mon-image', 'children'),
              [Input('bouton-chargement', 'contents')])

def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        if 'image' in content_type:
            image = Image.open(io.BytesIO(base64.b64decode(content_string)))
            predicted_class = classify_image(image, DNN)[0][0]
            proba_panneaux = classify_image(image, DNN)[1]
            nb_class=np.arange(1,44)
            return html.Div([
                html.Img(src=contents,style={'height':'15%','width':'15%','marginLeft': '550px', 'marginRight': 'auto','borderRadius': '5px'}),
                html.H3('Il y a {} % de chances que le panneau soit de type {}'.format(round((proba_panneaux[predicted_class])*100,2),CLASSES[predicted_class])),
                dcc.Graph(id='coucou', figure=px.bar(x=nb_class, y=proba_panneaux, barmode="group", title="Probabilités pour chaque panneaux",
                          labels={'x':'Type de panneaux', 'y':'Probabilité'})),
            ])
        else:
            try:
                # Décodage de l'image transmise en base 64 (cas des fichiers ppm)
                # fichier base 64 --> image PIL
                image = Image.open(io.BytesIO(base64.b64decode(content_string)))
                # image PIL --> conversion PNG --> buffer mémoire 
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                # buffer mémoire --> image base 64
                buffer.seek(0)
                img_bytes = buffer.read()
                content_string = base64.b64encode(img_bytes).decode('ascii')
                # Appel du modèle de classification
                predicted_class = classify_image(image, DNN)[0][0]
                proba_panneaux = classify_image(image, DNN)[1]
                nb_class=np.arange(1,44)
                # Affichage de l'image
                return html.Div([
                    html.Img(src='data:image/png;base64,' + content_string,style={'height':'15%','width':'15%','marginLeft': '550px', 'marginRight': 'auto','borderRadius': '5px'}),
                    html.H3('Il y a {} % de chances que le panneau soit de type {}'.format(round((proba_panneaux[predicted_class])*100,2),CLASSES[predicted_class])),
                    dcc.Graph(id='coucou', figure=px.bar(x=nb_class, y=proba_panneaux, barmode="group",title="Probabilités pour chaque panneaux",
                              labels={'x':'Type de panneaux', 'y':'Probabilité'})),
                ])
            except:
                return html.Div([
                    html.Div('Uniquement des images svp : {}'.format(content_type)),             
                    html.Div('Raw Content'),
                    html.Pre(contents, style=pre_style)
                ])
            

# Start the application
if __name__ == '__main__':
    app.run_server(debug=True)
