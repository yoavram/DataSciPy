import zipfile
import pickle

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

filename = '../data/tennis.zip'
player = 'Rafael-Nadal'
features = ['player1 aces', 'player1 double faults']
model_filename = 'logistic_tennis.model'

if __name__ == '__main__':
	tennis_zip = zipfile.ZipFile(filename)    
	path = os.path.join('data', '{}.csv')
	path = path.format(player.replace(' ', '-'))
	with tennis_zip.open(path) as f:
	    df = pd.read_csv(f)

	df['win'] = df['player1 name'] == df['winner']
	target = 'win'
	idx = np.isfinite(df[features]).all(axis=1)
	df_ = df[idx]

	X_train, X_test, y_train, y_test = model_selection.train_test_split(
		df_[features], df_[target], test_size=0.75)

	model = LogisticRegression()
	model.fit(X_train, y_train)
	print("Accuracy:", model.score(X_test, y_test))
	with open(model_filename, 'wb') as f:
		pickle.dump(model, f)
		print("Saving model to", model_filename)
	print('Prediction:')
	print(X_test.iloc[0], y_test.iloc[0])
