import csv
import os
import numpy as np
from shutil import copyfile

from .CSVReader import CSVReader
from .TxtManipulator import TxtManipulator

class FastTextPreprocessor:

	@classmethod
	def convert_csv_to_fasttext_input(cls, input_csv, output_txt, data_columns, 
									  label_column, label_prefix="__label__", 
									  append_label_prefix=False):
		file = CSVReader.csv_to_numpy_list(input_csv)
		
		data = 	file[:,data_columns]
		if append_label_prefix:
			label_column = np.array([label_prefix+str(row[label_column]) for row in file])
		else:
			label_column = file[:, label_column]

		output_data = np.array([label_column[index]+' , '+' '.join(str(col) for col in row) 
					  for index, row in enumerate(data)])
		TxtManipulator.write_txt_data_from_1d_np_list(output_txt, output_data)
