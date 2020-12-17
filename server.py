from flask import Flask, json, jsonify, url_for, send_from_directory, request, render_template, send_file

from PIL import Image
# import Image
import os
from werkzeug.utils import secure_filename
import sys
from flask_cors import CORS

HOST_URL_PORT='http://localhost:5000/'
app = Flask(__name__)

import cv2
import numpy as np


PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

def check_image(saved_path):
    print(saved_path)

    #== Parameters =======================================================================
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0,0.0,1.0) # In BGR format


    #== Processing =======================================================================

    #-- Read image -----------------------------------------------------------------------
    img = cv2.imread(saved_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Previously, for a previous version of cv2, this line was: 
    #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    #-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    #-- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    img         = img.astype('float32') / 255.0                 #  for easy blending

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

    cv2.imshow('img', masked)                                   # Display
    img = Image.fromarray(masked, 'RGB')
    result_file = secure_filename('my.png')
    img.save(result_file)
    return result_file

@app.route('/task/<task>', methods=['GET', 'POST'])
def task(task):
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        print(img)
        print(type(img))
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)
        filename = check_image(saved_path)
        return send_file(filename, mimetype='image/png')

    else:
        return 'Where is image?'


@app.route('/', methods=['GET'])
def hello_world():
	#print('--------------------- GET request ------------------', file=sys.stderr)
	return """
			<html>
				<head>
					<title>Remove Background</title>
                    <link rel="stylesheet" href="{}">
				</head>
				<body>
					<form method="POST" action="/task/compare" enctype=multipart/form-data>
						Select Image File<input type="file" name="image"><br /><br />
						<input type="submit" value="Submit"><br />
					</form>
                    <hr>
				</body>
			</html>
			""".format(url_for('static', filename='style.css'))

if __name__ == '__main__':
    cors = CORS(app, resources={r"*": {"origins": "*"}})
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', debug=True)
