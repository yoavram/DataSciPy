import urllib.request
import zipfile
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats


fname = '../data/anage_data.txt'
max_mass = 3e5

if __name__ == '__main__':
	df = pd.read_table(fname)
	df = df[df['Body mass (g)'] < max_mass]
	df['Body mass ^ 3/4'] = df['Body mass (g)'] ** (3/4)
	print('scatterplot...')
	sns.regplot('Body mass ^ 3/4', 'Metabolic rate (W)', data=df)
	plt.show()

	print('lmplot...')
	sns.lmplot(x='Body mass ^ 3/4', y='Metabolic rate (W)', hue='Class', data=df, col='Class', 
		sharex=False, sharey=False, scatter_kws=dict(alpha=0.3))
	plt.ylabel('Metabolic rate (W)')
	plt.show()

	print('linear regression...')
	for clazz, grp in df.groupby('Class'):
	    res = scipy.stats.linregress(grp['Body mass ^ 3/4'], grp['Metabolic rate (W)'])
	    print("{:s}: W = {:.2g} * g + {:.2g}".format(clazz, res.slope, res.intercept))
	