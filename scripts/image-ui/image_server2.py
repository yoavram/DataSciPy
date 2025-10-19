import os
import shutil
import uuid
import gzip
import io
from functools import wraps

from flask import Flask, jsonify, send_file, request, url_for, abort, after_this_request
from flask.ext.httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

from segment import open_image, save_image, segment_image


# configuration
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'images/')
EXTENSION = os.environ.get('EXTENSION',' .jpg')
SECRET_KEY = os.environ.get('SECRET_KEY', 'lucyintheskywithdiamonds')
PASSWORD_HASH = os.environ.get('PASSWORD_HASH', 'pbkdf2:sha1:1000$QaeYEdbL$83fd8e594b62385a4a7d28ee1a2bd2f3b98d2dcf')
TOKEN_EXPIRATION = int(os.environ.get('TOKEN_EXPIRATION', 3600)) # in secs


# init app
app = Flask(__name__)
app.config.from_object(__name__)
auth = HTTPBasicAuth()


if os.path.exists(app.config['UPLOAD_FOLDER']):
    shutil.rmtree(app.config['UPLOAD_FOLDER'])
os.mkdir(app.config['UPLOAD_FOLDER'])


# helpers
def generate_id():
	return str(uuid.uuid4())

def generate_path(_id):
	return os.path.join(app.config['UPLOAD_FOLDER'], "{}{}".format(_id, app.config['EXTENSION']))

def validate_path(path):
	if not os.path.exists(path):
		abort(404) 

def generate_url(_id):
	return url_for('get_image', image_id=_id, _external=True)


# authentication
@auth.verify_password
def verify_password(token, password):
	return verify_auth_token(token) or \
		check_password_hash(app.config['PASSWORD_HASH'], password)

def generate_auth_token():
    s = Serializer(app.config['SECRET_KEY'], expires_in=app.config['TOKEN_EXPIRATION'])
    return s.dumps({'verified': True}).decode('ascii')

def verify_auth_token(token):
	s = Serializer(app.config['SECRET_KEY'])
	try:
		data = s.loads(token)
	except:
		return False	
	return True


# error handlers
@app.errorhandler(404)
def not_found(error):
	return jsonify(error='not found'), 404

@app.errorhandler(400)
def bad_request(error):
	return jsonify(error='bad request'), 400

@app.errorhandler(405)
def not_allowed(error):
	return jsonify(error='method not allowed'), 405

@auth.error_handler
def unauthorized():
	return jsonify(error='unauthorized access'), 403

# compression: http://flask.pocoo.org/snippets/122/ 
def gzipped(f):
    @wraps(f)
    def view_func(*args, **kwargs):
        @after_this_request
        def zipper(response):
            accept_encoding = request.headers.get('Accept-Encoding', '')

            if 'gzip' not in accept_encoding.lower():
                return response

            response.direct_passthrough = False

            if (response.status_code < 200 or
                response.status_code >= 300 or
                'Content-Encoding' in response.headers):
                return response
            gzip_buffer = io.BytesIO()
            gzip_file = gzip.GzipFile(mode='wb', 
                                      fileobj=gzip_buffer)
            gzip_file.write(response.data)
            gzip_file.close()

            response.data = gzip_buffer.getvalue()
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Vary'] = 'Accept-Encoding'
            response.headers['Content-Length'] = len(response.data)

            return response

        return f(*args, **kwargs)

    return view_func


# views
@app.route('/api/2/ping', methods=['GET'])
def ping():
	return jsonify(result=True)

@app.route('/api/2/pingauth', methods=['GET'])
@auth.login_required
def pingauth():
	return jsonify(result=True)

@app.route('/api/2/token', methods=['GET'])
@auth.login_required
def get_auth_token():	
	token = generate_auth_token()
	return jsonify(token=token, expiration=app.config['TOKEN_EXPIRATION'])

@app.route('/api/2/image/<string:image_id>')
@auth.login_required
@gzipped
def get_image(image_id):
	image_path = generate_path(image_id)
	validate_path(image_path)
	return send_file(image_path)

@app.route('/api/2/image', methods=['POST'])
@auth.login_required
def post_image():
	file = request.files['file']
	image_id = generate_id()
	file.save(generate_path(image_id))
	return jsonify(image_id=image_id, url=generate_url(image_id))

@app.route('/api/2/image/<string:image_id>/segment')
@auth.login_required
def segment(image_id):
	image_path = generate_path(image_id)
	validate_path(image_path)
	image = open_image(image_path)
	segmented = segment_image(image)
	segmented_image_id = generate_id()
	segmented_image_path = generate_path(segmented_image_id)
	save_image(segmented_image_path, segmented)
	return jsonify(image_id=segmented_image_id, url=generate_url(segmented_image_id))


if __name__=='__main__':
	app.run(host='127.0.0.1', port=2000, debug=True)
	app.logger.info("Server shuting down")