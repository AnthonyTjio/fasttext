import sys
import getopt
import numpy as np

from driver import Driver

command = sys.argv[1].lower()

if(command == "convert"):
	opts, args = getopt.getopt(sys.argv[2:], "i:o:d:l:p:", ["input=", "output=", "prefix=", 
															"data_column=", "label_column="])
	input_file = 'data.csv'
	output_file = 'training.txt'
	prefix = False
	label_prefix = None
	data_column = []
	label_column = 3
	
	for opt, arg in opts:
		if opt in ('-i', '--input'):
			input_file = arg
		elif opt in ('-o', 'output'):
			output_file = arg
		elif opt in ('-d', '--data_column'):
			data_column.append(int(arg))
		elif opt in ('-l', '--label_column'):
			label_column = int(arg)
		elif opt in ('-p', '--prefix'):
			label_prefix = arg

	if label_prefix:
		prefix = True

	Driver.convert_dataset(input_csv=input_file, output_txt=output_file, data_columns=data_column, 
						   label_column=label_column, label_prefix=label_prefix, 
						   append_label_prefix=prefix)
	
elif(command == "train"):
	opts, args = getopt.getopt(sys.argv[2:], "i:c:", ["input=", "config="])

	input_file = 'training.txt'
	config = 'config.json'

	for opt, arg in opts:
		if opt in ('-i', '--input'):
			input_file = arg
		elif opt in ('-c', '--config'):
			config = arg

	Driver.train_model(input_file, config)

elif(command == 'test'):
	opts, args = getopt.getopt(sys.argv[2:], "i:c:", ["input=", "config="])

	input_file = 'test.txt'
	config = 'config.json'

	for opt, arg in opts:
		if opt in ('-i', '--input'):
			input_file = arg
		elif opt in ('-c', '--config'):
			config = arg

	Driver.test_model(input_file, config)

elif(command == 'predict'):
	opts, args = getopt.getopt(sys.argv[2:], "i:c:f:o:", ["input=", "config=", "file=", "output="])

	texts = []
	config = 'config.json'
	file = None
	output = None

	for opt, arg in opts:
		if opt in ('-i', '--input'):
			texts.append(arg)
		elif opt in ('-c', '--config'):
			config = arg
		elif opt in ('-f', '--file'):
			file = arg
		elif opt in ('-o', '--output'):
			output = arg

	if file:
		Driver.predict_txt_to_csv(file, output, config)
	else:
		Driver.predict_list(texts, config)