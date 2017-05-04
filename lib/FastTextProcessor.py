import fasttext
import os 
import json

class FastTextProcessor:

	model=None
	dim= None
	learning_rate= None
	epoch= None
	min_count= None
	word_ngrams= None
	bucket= None
	thread= None
	silent= None
	label_prefix= None

	def __init__(self, config_file="./config.json"):
		with open(config_file, 'r') as conf:
			config = json.load(conf)
			self.model = config['model']
			self.dim = config['dim']
			self.learning_rate = config['learning_rate']
			self.epoch = config['epoch']
			self.min_count = config['min_count']
			self.word_nrgams = config['word_ngrams']
			self.bucket = config['bucket']
			self.thread = config['thread']
			self.silent = config['silent']
			self.label_prefix = config['label_prefix']
			try:
				self.classifier = fasttext.load_model(self.model+'.bin', label_prefix=self.label_prefix)
			except Exception as ex:
				self.classifier = None
				print("Model not found!")

	def train(self, training_data):
		self.classifier = fasttext.supervised(training_data, self.model, dim=self.dim, lr=self.learning_rate,
											  epoch=self.epoch, min_count=self.min_count, 
											  word_ngrams=self.word_ngrams, bucket=self.bucket, 
											  thread = self.thread, silent=self.silent)

	def test(self, test_data):
		result = self.classifier.test(test_data)
		return result

	def predict(self, predict_data):
		result = self.classifier.predict(predict_data)
		return result