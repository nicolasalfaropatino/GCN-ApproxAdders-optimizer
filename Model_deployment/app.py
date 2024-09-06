import numpy as np
from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename
from main import getGraph
import os
import pickle
from torch_geometric.nn import GCNConv
import torch

UPLOAD_FOLDER = 'static/graphs/'

app = Flask(__name__, static_folder="static")

app.secret_key = "secret key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            getGraph(filename)
            graph = getGraph(filename)
            flash(graph)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash(full_filename)
            return redirect('/')

if __name__ == "__main__":
  app.run()