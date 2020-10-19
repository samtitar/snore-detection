from flask import Flask, render_template, request, redirect, jsonify, session
from flask_socketio import SocketIO, emit

import torch
import torch.nn as nn
import torch.nn.functional as F

import io
import os
import json
import librosa
import numpy as np

N_FEATURES = 20
S_LENGTH = 17

class Classifier(nn.Module):
    def __init__(self, n_features=32):
        super(Classifier, self).__init__()
        
        self.lin1 = nn.Linear(n_features, 50)
        self.lin2 = nn.Linear(50, 60)
        self.lstm = nn.LSTM(60, 30, bidirectional=True, batch_first=True)
        self.lin3 = nn.Linear(60, 1)
    
    def forward(self, x, hidden):
        # Create mind vector and run trough LSTM
        y = F.relu(self.lin1(x))
        y = F.relu(self.lin2(y))
        y, hidden = self.lstm(y, hidden)
        
        # Get last prediction from LSTM
        y = y[:, -1, :]
        
        # Classify
        return torch.sigmoid(self.lin3(y)), hidden

model = Classifier(n_features=N_FEATURES)
model.eval()

app = Flask(__name__)
app.secret_key = 'snore_detector_v1'
socketio = SocketIO(app)
connections = []

def get_features(signal):
    return librosa.feature.mfcc(signal, n_mfcc=N_FEATURES).T

def get_prediction(features, lstm_state):
    with torch.no_grad():
        prediction, lstm_state = model(features, lstm_state)
    return prediction.item(), lstm_state

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@socketio.on('connect')
def connect():
    connection = {}

    # Initialize session hidden and cell state for client
    connection['hidd'] = torch.zeros(2, 1, 30)
    connection['cell'] = torch.zeros(2, 1, 30)

    connections.append(connection)
    emit('client_id',  { 'id': len(connections) - 1 })

@socketio.on('predict')
def predict(signal, client_id):
    '''Process audio signal segment'''
    signal = list(zip(*signal.items()))[1] # JS array to python list
    features = get_features(np.array(signal).astype(float))
    features = torch.tensor(features, dtype=torch.float)
    features = features[:S_LENGTH].view(1, S_LENGTH, N_FEATURES)

    # Get clients LSTM state
    lstm_state = (connections[client_id]['hidd'],
                  connections[client_id]['cell'])
    
    prediction, lstm_state = get_prediction(features, lstm_state)

    # Update clients LSTM state
    connections[client_id]['hidd'] = lstm_state[0]
    connections[client_id]['cell'] = lstm_state[1]

    emit('prediction', { 'confidence': prediction })

if __name__ == '__main__':
    model_state = torch.load('model.pt')
    model.load_state_dict(model_state)
    app.run(host='0.0.0.0')
    socketio.run(app)