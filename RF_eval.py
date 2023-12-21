import numpy as np
import pandas as pd
from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sktime.datatypes._panel._convert import from_2d_array_to_nested
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction import DictVectorizer
import cv2

eval_files = sorted(os.listdir("落點偵測datasets/test/"))
for eval_file in eval_files:
	print('eval_file ',eval_file)
	eval_csv = pd.read_csv("落點偵測datasets/test/" + eval_file)

	In = pd.DataFrame([eval_csv["x"],eval_csv["y"],eval_csv["V"]]).T

	for i in range(5, 0, -1): 
		In[f"X_f{i}"] = In["x"].shift(i, fill_value=0)
		In[f"X_b{i}"] = In["x"].shift(-i, fill_value=0)
	for i in range(5, 0, -1):
		In[f"Y_f{i}"] = In["y"].shift(i, fill_value=0)
		In[f"Y_b{i}"] = In["y"].shift(-i, fill_value=0)
	for i in range(5, 0, -1): 
		In[f"V_f{i}"] = In["V"].shift(i, fill_value=0)
		In[f"V_b{i}"] = In["V"].shift(-i, fill_value=0)

	Input = In[5:-5].set_index(i for i in range(len(In.index)-10))  #捨棄前5幅和後5幅

	Xs = Input[['X_f5','X_f4','X_f3','X_f2','X_f1','x','X_b1','X_b2','X_b3','X_b4','X_b5']]

	Ys = Input[['Y_f5', 'Y_f4', 'Y_f3', 'Y_f2', 'Y_f1', 'y', 'Y_b1', 'Y_b2', 'Y_b3', 'Y_b4', 'Y_b5']]

	Vs = Input[['V_f5', 'V_f4', 'V_f3', 'V_f2', 'V_f1', 'V', 'V_b1', 'V_b2', 'V_b3', 'V_b4', 'V_b5']]

	Input = pd.concat([Xs,Ys,Vs],axis=1)

	gt = pd.DataFrame([eval_csv["bounce"]]).T
	gt = gt[5:-5].set_index(i for i in range(len(gt.index)-10))  #捨棄前5幅和後5幅

	groundTruth = list(np.where(gt == 1)[0])
	#載入訓練好的模型
	with open('best_RF.pkl','rb') as f:
		model = pickle.load(f)

	pred = model.predict(Input)
	pred = pd.DataFrame({'bounce':pred})	

	predictions = list(np.where(pred == 1)[0])

	#計算混淆矩陣			
	tn, fp, fn, tp = 0, 0, 0, 0
	for i in range(len(pred.index)):
		if gt.at[i,'bounce'] == 1 and pred.at[i,'bounce'] == 1:
			tp+=1
		if gt.at[i,'bounce'] == 1 and pred.at[i,'bounce'] == 0:
			fn+=1
		if gt.at[i,'bounce'] == 0 and pred.at[i,'bounce'] == 0:
			tn+=1
		if gt.at[i,'bounce'] == 0 and pred.at[i,'bounce'] == 1:
			fp+=1
	print('tn: ',tn,'fp: ',fp,'fn: ',fn,'tp: ',tp)
	# 績效
	accuracy = metrics.accuracy_score(gt, pred)
	print('accuracy ',accuracy)
	precision = metrics.precision_score(gt, pred)
	print('precision ',precision)
	recall = metrics.recall_score(gt, pred)
	print('recall ',recall)
	f1 = metrics.f1_score(gt, pred)
	print('f1 ',f1)
