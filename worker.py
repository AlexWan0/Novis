from rq import Queue, Worker, Connection
import redis
import detection

with Connection():
	qs = ['default']

	detection.init_blood_model()

	w = Worker(qs)
	w.work()