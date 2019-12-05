import numpy as np
from keras.utils import Sequence, to_categorical

class DataGenerator(Sequence):
	def __init__(self, x_data, y_data, num_classes, batch_size=32, shuffle=True):
		self.x_data = x_data
		self.y_data = y_data
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		return int(np.ceil(len(self.y_data) / self.batch_size))

	def __getitem__(self, index):
		batch_x = self.x_data[index*self.batch_size:(index+1)*self.batch_size]
		batch_y = self.y_data[index*self.batch_size:(index+1)*self.batch_size]
		return batch_x, to_categorical(batch_y, self.num_classes)

	def on_epoch_end(self):
		if self.shuffle:
			shuffled = np.random.permutation(range(len(self.y_data)))
			self.x_data = self.x_data[shuffled]
			self.y_data = self.y_data[shuffled]
