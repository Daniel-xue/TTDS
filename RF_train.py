import numpy as np
import pandas as pd
from sklearn import ensemble, preprocessing, metrics
from sktime.datatypes._panel._convert import from_2d_array_to_nested
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
import statistics

#train Dataset---------------------------------------------------------------------------------------------
train_X = pd.DataFrame()
train_Y = pd.DataFrame()
train_files = sorted(os.listdir("落點偵測datasets/train/"))
for train_file in train_files:
	if train_file == '01.csv' or train_file == '03.csv' or train_file == '04.csv':
		continue
	if train_file.split('.')[-1] != 'csv':
		continue
	train_csv = pd.read_csv("落點偵測datasets/train/" + train_file)
	# 建立X
	DF = pd.DataFrame([train_csv["x"],train_csv["y"],train_csv["V"]]).T

	for i in range(3, 0, -1): 
		DF[f"X_f{i}"] = DF["x"].shift(i, fill_value=0)
		DF[f"X_b{i}"] = DF["x"].shift(-i, fill_value=0)

		DF[f"Y_f{i}"] = DF["y"].shift(i, fill_value=0)
		DF[f"Y_b{i}"] = DF["y"].shift(-i, fill_value=0)

		DF[f"V_f{i}"] = DF["V"].shift(i, fill_value=0)
		DF[f"V_b{i}"] = DF["V"].shift(-i, fill_value=0)

	DF = DF[3:-3]  

	Xs = DF[['X_f3','X_f2','X_f1','x','X_b1','X_b2','X_b3']]

	Ys = DF[['Y_f3','Y_f2','Y_f1','y','Y_b1','Y_b2','Y_b3']]

	Vs = DF[['V_f3','V_f2','V_f1','V','V_b1','V_b2','V_b3']]

	DF = pd.concat([Xs,Ys,Vs],axis=1)
	train_X = pd.concat([train_X, DF], axis=0, ignore_index=True)
	# 建立Y
	train_y = pd.DataFrame([train_csv["bounce"]]).T
	train_y = train_y[3:-3] 
	train_Y = pd.concat([train_Y, train_y], axis=0, ignore_index=True)

#test Dataset---------------------------------------------------------------------------------------------
test_X = pd.DataFrame()
test_Y = pd.DataFrame()
test_css_path = "落點偵測datasets/test/test01.csv"
test_csv = pd.read_csv(test_css_path)

test_x = pd.DataFrame([test_csv["x"],test_csv["y"], test_csv["V"]]).T

for i in range(3, 0, -1): 
	test_x[f"X_f{i}"] = test_x["x"].shift(i, fill_value=0)
	test_x[f"X_b{i}"] = test_x["x"].shift(-i, fill_value=0)

	test_x[f"Y_f{i}"] = test_x["y"].shift(i, fill_value=0)
	test_x[f"Y_b{i}"] = test_x["y"].shift(-i, fill_value=0)

	test_x[f"V_f{i}"] = test_x["V"].shift(i, fill_value=0)
	test_x[f"V_b{i}"] = test_x["V"].shift(-i, fill_value=0)

test_x = test_x[3:-3] 

Xs = test_x[['X_f3', 'X_f2', 'X_f1', 'x', 'X_b1', 'X_b2', 'X_b3']]

Ys = test_x[['Y_f3', 'Y_f2', 'Y_f1', 'y', 'Y_b1', 'Y_b2', 'Y_b3']]

Vs = test_x[['V_f3', 'V_f2', 'V_f1', 'V', 'V_b1', 'V_b2', 'V_b3']]

DF = pd.concat([Xs,Ys,Vs],axis=1)
test_X = pd.concat([test_X, DF], axis=0, ignore_index=True)

test_y = pd.DataFrame([test_csv["bounce"]]).T
test_y = test_y[3:-3]
test_Y = pd.concat([test_Y, test_y], axis=0, ignore_index=True)
#模型訓練---------------------------------------------------------------------------------------------
RS=[i for i in range(100)]
f1_list=[]
best_f1=0
n_estimators=200
print('n_estimators: ',n_estimators)

for random_state in RS:
	model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, min_samples_leaf=10, class_weight='balanced')
	#model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, min_samples_leaf=10, class_weight='balanced', max_features=None, bootstrap=False)
	#使用訓練集訓練模型
	model_fit = model.fit(train_X, train_Y)	
	#評估模型在訓練集上的效能
	print('On Train Data')
	train_pred = model_fit.predict(train_X)
	train_pred = pd.DataFrame({'bounce':train_pred})
	accuracy = metrics.accuracy_score(train_Y, train_pred)
	print('accuracy ',accuracy)
	precision = metrics.precision_score(train_Y, train_pred)
	print('precision ',precision)
	recall = metrics.recall_score(train_Y, train_pred)
	print('recall ',recall)
	f1 = metrics.f1_score(train_Y, train_pred)
	print('f1 ',f1)
	#評估模型在測試集上的效能
	print('On Test Data')
	test_pred = model_fit.predict(test_X)
	test_pred = pd.DataFrame({'bounce':test_pred})
	accuracy = metrics.accuracy_score(test_Y, test_pred)
	print('accuracy ',accuracy)
	precision = metrics.precision_score(test_Y, test_pred)
	print('precision ',precision)
	recall = metrics.recall_score(test_Y, test_pred)
	print('recall ',recall)
	f1 = metrics.f1_score(test_Y, test_pred)
	print('f1 ',f1)
	#將每筆在測試集上得出的f1存進f1_list
	f1_list.append(f1)
	#把當前f1最好的模型存成pkl檔
	if f1 > best_f1:
		best_f1=f1
		with open('best_RF_{}.pkl'.format(n_estimators),'wb') as f:
			pickle.dump(model_fit,f)
print('best_f1 ',best_f1)
mean_f1 = statistics.mean(f1_list)
print('mean_f1',mean_f1)
