import os
import shutil
import uuid

from flask import Flask, jsonify, send_file, request, url_for

from segment import open_image, save_image, segment_image


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images/'
app.config['EXTENSION'] = '.jpg'

if os.path.exists(app.config['UPLOAD_FOLDER']):
    shutil.rmtree(app.config['UPLOAD_FOLDER'])
os.mkdir(app.config['UPLOAD_FOLDER'])

def generate_id():
	return str(uuid.uuid4())

def generate_path(_id):
	return os.path.join(app.config['UPLOAD_FOLDER'], "{}{}".format(_id, app.config['EXTENSION']))

def generate_url(_id):
	return url_for('get_image', image_id=_id, _external=True)

@app.route('/api/1/image/<string:image_id>')
def get_image(image_id):
	return send_file(generate_path(image_id))


@app.route('/api/1/image', methods=['POST'])
def post_image():
	file = request.files['file']
	image_id = generate_id()
	file.save(generate_path(image_id))
	return jsonify(image_id=image_id, url=generate_url(image_id))


@app.route('/api/1/image/<string:image_id>/segment')
def segment(image_id):
	image_path = generate_path(image_id)
	image = open_image(image_path)
	segmented = segment_image(image)
	segmented_image_id = generate_id()
	segmented_image_path = generate_path(segmented_image_id)
	save_image(segmented_image_path, segmented)
	return jsonify(image_id=segmented_image_id, url=generate_url(segmented_image_id))


if __name__=='__main__':
	app.run(host='127.0.0.1', port=5000, debug=True)
	app.logger.info("Server shuting down")