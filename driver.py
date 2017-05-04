import fasttext
import json
import os
import numpy as np

from lib.FastTextPreprocessor import FastTextPreprocessor
from lib.FastTextProcessor import FastTextProcessor

from lib.CSVManipulator import CSVManipulator
from lib.TxtReader import TxtReader

class Driver:
	def ensure_dependencies():
		default_model_dir = "./model"
		if not os.path.exists(default_model_dir):
			os.makedirs(default_model_dir)

	ensure_dependencies()

	@classmethod
	def convert_dataset(cls, input_csv, output_txt, data_columns, 
									  label_column, label_prefix="__label__", 
									  append_label_prefix=False):
		FastTextPreprocessor.convert_csv_to_fasttext_input(input_csv, output_txt, data_columns, 
														   label_column, label_prefix, append_label_prefix)

	@classmethod
	def train_model(cls, training_data, config_file='./config.json'):
		fasttextProcessor = FastTextProcessor(config_file)
		fasttextProcessor.train(training_data)

	@classmethod
	def test_model(cls, test_data, config_file='./config.json'):
		fasttextProcessor = FastTextProcessor(config_file)
		result = fasttextProcessor.test(test_data)

		print('P@1:', result.precision)
		print('R@1:', result.recall)
		print('Number of examples:', result.nexamples)
		
	@classmethod
	def predict_list(cls, texts, config_file='./config.json'):
		fasttextProcessor = FastTextProcessor(config_file)
		labels = fasttextProcessor.predict(texts)

		print(labels)

	@classmethod
	def predict_txt_to_csv(cls, txt_file, csv_file, config_file="./config.json"):
		fasttextProcessor = FastTextProcessor(config_file)
		
		texts = TxtReader.read_txt_to_numpy_list(txt_file)
		labels = np.array(fasttextProcessor.predict(texts))

		texts = np.array([texts]).T

		print(texts.shape)
		print(labels.shape)

		csv_input = np.concatenate( (texts, labels), axis=1)
		CSVManipulator.write_csv_data_from_2d_np_list(csv_file, csv_input)




