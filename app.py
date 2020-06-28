from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import base64
from rq import Queue, Connection
from rq.job import Job
from redis import Redis
import detection
import time

app = Flask(__name__, static_folder='static')

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024

q = Queue(connection=Redis('localhost', 6379))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
	data = request.form

	if 'file' not in request.files:
		return {'success': False, 'message': 'No file'}

	file = request.files['file']

	print('new request')

	if file.filename == '':
		return {'success': False, 'message': 'No file'}

	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		image_string = base64.b64encode(file.read())

		job = q.enqueue_call(func=detection.blood_detect, args=(image_string,), result_ttl=800)

		while not job.is_finished and not job.is_failed:
			time.sleep(2)
		
		return job_done(job)
	else:
		return {'success': False, 'message': 'Invalid file extension'}

	return {'success': False, 'message': 'Unknown error'}

def job_done(job):
	if job.is_finished:
		result = job.result

		print('finished')
		return jsonify(result)
	else:
		print('failed')

		return {'success': False, 'message': 'Unknown error'}

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80)