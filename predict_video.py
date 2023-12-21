import argparse
import queue
import pandas as pd
import pickle
import imutils
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
import sys
import time
from tensorflow import keras
from court_detector import CourtDetector
from Models.trackNet import ResNet_Track
from TrackPlayers.trackplayers import *
from utils import get_video_properties
from detection import *
from pickle import load
from focal_loss import BinaryFocalLoss
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from collections import deque
import warnings
from sklearn import metrics
from detection import create_top_view

warnings.simplefilter(action='ignore', category=FutureWarning)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# parse parameters
parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path", type=str, default='VideoInput/test01.mp4')
parser.add_argument("--output_video_path", type=str, default='VideoOutput/test01.mp4')
parser.add_argument("--gt_csv_path", type=str, default='落點偵測datasets/test/test01.csv')
parser.add_argument("--minimap", type=bool, default=True)
parser.add_argument("--bounce", type=bool, default=True)
parser.add_argument("--crop_width", type=int, default=400)
parser.add_argument("--crop_height", type=int, default=200)

args = parser.parse_args()

gt_csv_path = args.gt_csv_path
input_video_path = args.input_video_path
video_name = (input_video_path.split('.')[0]).split('/')[-1]
print('video_name',video_name)
output_video_path = args.output_video_path
minimap = args.minimap
bounce = args.bounce
Tracknet_weights_path = 'WeightsTracknet/TrackNet2'

# load TrackNet model & weights
width, height = 512, 288
BATCH_SIZE = 1
FRAME_STACK = 3

opt = keras.optimizers.Adadelta(learning_rate=1.0)
model=ResNet_Track(input_shape=(3, height, width))
model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=opt, metrics=[keras.metrics.BinaryAccuracy()])
try:
	model.load_weights(Tracknet_weights_path)
	print("Load weights successfully")
except:
	print("Fail to load weights, please modify path in parser.py --load_weights")

# In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
q = queue.deque()
for _ in range(0, 8):
	q.appendleft(None)

#存取每一幅中預測出的桌球的位置作標
coords = []
crop_coords = []
gray_imgs = deque()# gray imgs deque

# court
court_detector = CourtDetector()

# get videos properties
fps, length, v_width, v_height = get_video_properties(cap)
ratio = v_height / height
print('ratio: ',ratio)
size = ((v_width, v_height))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')#build video
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, size)#video paramter

#將前FRAME_STACK幅圖片合併,已成為trackNet的輸入
for i in range(FRAME_STACK):
	success, frame = cap.read()#read video to img
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#transform gray
	gray = np.expand_dims(gray, axis=2)
	gray_imgs.append(gray)#append deque

time_list=[]  #ball detect 個別花費時間
court_detect_time=0  #court detect 花費時間
#ball detect & court detect
pre_pt = 0
while success:
	#球桌（框線）預測
	start = time.time()
	#!!!由於球桌一定會出現在影片的中央,將偵測範圍縮小以提升偵測速度及精度
	frame = frame[200:500,400:900]
	if pre_pt == 0:
		points = court_detector.detect_table(frame)
	else:
		points = court_detector.track_table(frame)

	end = time.time()
	court_detect_time+=(end-start)

	if points is None:
		print('not found the table')
		pre_pt = 0
		pass
	else:
		for i in range(0, len(points), 4):
			x1, y1, x2, y2 = points[i], points[i+1], points[i+2], points[i+3]
			cv2.line(frame, (x1+args.crop_height,y1+args.crop_width),(x2+args.height,y2+args.crop_crop_width), (0,0,255), 3)
	#桌球（點）預測
	img_input = np.concatenate(gray_imgs, axis=2)#three gray imgs gray_imgs
	img_input = cv2.resize(img_input, (width, height))
	img_input = np.moveaxis(img_input, -1, 0)
	img_input = np.expand_dims(img_input, axis=0)
	img_input = img_input.astype('float')/255.#reduce quantity of computation

	start = time.time()
	y_pred = model.predict(img_input, batch_size=BATCH_SIZE)#feed in model
	end = time.time()
	time_list.append(end-start)
	y_pred = y_pred > 0.5#have ball >0.5
	y_pred = y_pred.astype('float32')
	h_pred = y_pred[0]*255
	h_pred = h_pred.astype('uint8')

	if np.amax(h_pred) <= 0:#no ball
		#print('not found the ball')
		coords.append([None,None])
		crop_coords.append([None,None])
		output_video.write(frame)

	else:# yes ball
		cnts, _ = cv2.findContours(h_pred[0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#search ball contour
		rects = [cv2.boundingRect(ctr) for ctr in cnts]#visualize ball contour ROI
		max_area_idx = 0
		max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
		for i in range(1, len(rects)):#get ball coordinate position
			area = rects[i][2] * rects[i][3]
			if area > max_area:
				max_area_idx = i
				max_area = area
		target = rects[max_area_idx]

		(cx_pred, cy_pred) = (int(ratio*(target[0] + target[2] / 2)), int(ratio*(target[1] + target[3] / 2)))
		cv2.circle(frame, (cx_pred, cy_pred), 5, (0,255,0), -1)#draw the ball position
		coords.append([cx_pred,cy_pred])
		crop_coords.append([cx_pred - args.crop_width, cy_pred - args.crop_height])
		output_video.write(frame)#will img write video

	success, frame = cap.read()#read  next img
	if success:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#transform gray
		gray = np.expand_dims(gray, axis=2)
		gray_imgs.append(gray)
		gray_imgs.popleft()#redues gray img deque

print('court_detect_time ',court_detect_time)
output_video.release()
#-------------------------------------------------------------------------------------------
if minimap == True:
	game_video = cv2.VideoCapture(output_video_path)

	fps, length, v_width, v_height = get_video_properties(game_video)
	output_video = cv2.VideoWriter('VideoOutput/video_with_map_{}.mp4'.format(video_name), fourcc, fps, (v_width, v_height))
	print('Adding the mini-map...')
	transforms = create_top_view(court_detector, crop_coords, fps)
	minimap_video = cv2.VideoCapture('VideoOutput/minimap.mp4')

	while True:
		ret1, frame1 = game_video.read()
		ret2, frame2 = minimap_video.read()

		if ret1 and ret2:
			output = merge(frame1, frame2)
			output_video.write(output)
		else:
			break
		game_video.release()
		minimap_video.release()
		output_video.release()
#-------------------------------------------------------------------------------------------
if bounce == True:
	xy = pd.DataFrame({'x': [coord[0] for coord in coords[:]], 'y':[coord[1] for coord in coords[:]]})
	xy = xy.interpolate()
	xy = xy.fillna(0)
	#計算速度
	Vx = []
	Vy = []
	V = []
	for i in range(len(coords)-1):
		x1 = xy.at[i,'x']
		x2 = xy.at[i+1,'x']
		y1 = xy.at[i,'y']
		y2 = xy.at[i+1,'y']
		p1=[x1,y1]
		p2=[x2,y2]
		#print(p1,p2)

	x = (p1[0]-p2[0])*fps
	y = (p1[1]-p2[1])*fps
	Vx.append(x)
	Vy.append(y)
	for i in range(len(Vx)):
		vx = Vx[i]
		vy = Vy[i]

	v = (vx**2+vy**2)**0.5
	V.append(v)
	#將x,y,V包一起(最後一幅不採計,因為算不出V)
	xyv = xy.drop([len(coords)-1],axis=0)
	xyv.insert(2, column='V', value=V)

	for i in range(5, 0, -1):
		xyv[f"X_f{i}"] = xyv["x"].shift(i, fill_value=0)
		xyv[f"X_b{i}"] = xyv["x"].shift(i, fill_value=0)
	for i in range(5, 0, -1):
		xyv[f"Y_f{i}"] = xyv["y"].shift(i, fill_value=0)
		xyv[f"Y_b{i}"] = xyv["y"].shift(i, fill_value=0)
	for i in range(5, 0, -1):
		xyv[f"V_f{i}"] = xyv["V"].shift(i, fill_value=0)
		xyv[f"V_b{i}"] = xyv["V"].shift(i, fill_value=0)

	xyv = xyv[5:-5]
	Xs = xyv[['X_f5','X_f4','X_f3','X_f2','X_f1','x','X_b1','X_b2','X_b3','X_b4','X_b5']]
	Ys = xyv[['Y_f5', 'Y_f4', 'Y_f3', 'Y_f2', 'Y_f1', 'y', 'Y_b1', 'Y_b2', 'Y_b3', 'Y_b4', 'Y_b5']]
	Vs = xyv[['V_f5', 'V_f4', 'V_f3', 'V_f2', 'V_f1', 'V', 'V_b1', 'V_b2', 'V_b3', 'V_b4', 'V_b5']]

	X = pd.concat([Xs,Ys,Vs],1)

	# load the pre-trained classifier
	clf = pickle.load(open('best_RF_200.pkl', 'rb'))
	Y = clf.predict(X)
	pred = pd.DataFrame({'bounce':Y})

	gt_csv = pd.read_csv(gt_csv_path)
	gt = pd.DataFrame([gt_csv["bounce"]]).T
	gt = gt[5:-5].set_index(i for i in range(len(gt.index)-10))  #捨棄前5幅和後5幅

	print('pred\n',pred)
	print('gt\n',gt)
	assert len(pred.index) == len(gt.index),'len(pred.index) != len(gt.index)'
	groundTruth = list(np.where(gt == 1)[0])
	groundTruth = [x+7 for x in groundTruth]
	predictions = list(np.where(pred == 1)[0])
	predictions = [x+7 for x in predictions]

	if minimap == True:
		cap = cv2.VideoCapture('VideoOutput/video_with_map_{}.mp4'.format(video_name))
	else:
		cap = cv2.VideoCapture(output_video_path)

	fps, length, v_width, v_height = get_video_properties(cap)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	output_video = cv2.VideoWriter('VideoOutput/final_video_{}.mp4'.format(video_name), fourcc, fps, (output_width, output_height))

	# demo
	i = 1
	while True:
		ret, frame = cap.read()
		if not ret:break
		i += 1
		if i in predictions:
			if coords[i-2] != [None,None]:
				ball_pos = coords[i-2]
				cv2.circle(frame, (ball_pos[0], ball_pos[1]), 5, (255, 255, 255), -1)
				transform = transforms[i-2]
				print('transform: ',transform)
				cv2.putText(frame, '{}'.format(transform), (ball_pos[0],ball_pos[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

		if i in groundTruth:
			ball_pos = xyv.iat[(i-5),5], xyv.iat[(i-5),16]
			cv2.circle(frame, (ball_pos[0], ball_pos[1]), 5, (0, 255, 255), -1)
			inv_mats = court_detector.game_warp_matrix[i-2]
			transform = cv2.perspectiveTransform(ball_pos, inv_mats)[0][0].astype('int64')
			print('transform: ',transform)
			cv2.putText(frame, '{}'.format(transform), (ball_pos[0]-10,ball_pos[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

		output_video.write(frame)
		cv2.imwrite('VideoOutput/final_video_{}/{}.jpg'.format(video_name,i),frame)
		cap.release()
		output_video.release()

	# 績效
	accuracy = metrics.accuracy_score(gt, pred)
	print('accuracy ',accuracy)
	precision = metrics.precision_score(gt, pred)
	print('precision ',precision)
	recall = metrics.recall_score(gt, pred)
	print('recall ',recall)
	f1 = metrics.f1_score(gt, pred)
	print('f1 ',f1)
