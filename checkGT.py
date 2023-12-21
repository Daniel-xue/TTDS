import cv2
import pandas as pd
import numpy as np

video_name='test01'
print('video_name: ',video_name)
video = cv2.VideoCapture('VideoInput/{}.mp4'.format(video_name))
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
csv = pd.read_csv('落點偵測datasets/test/{}.csv'.format(video_name))
gt = pd.DataFrame([csv["bounce"]]).T
gt_list = list(np.where(gt == 1)[0])
gt_list = [x+2 for x in gt_list]

i = 0
while i<(n_frames-3):  
	ret, frame = video.read()    
	if not ret:break
	#get index for gt_list
	if i > 2:
		if i in gt_list:
			ball_pos = csv.at[(i-2),'x'], csv.at[(i-2),'y']
			cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 5, (0, 0, 255), -1)
		else:
			ball_pos = csv.at[(i-2),'x'], csv.at[(i-2),'y']
			cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 5, (255, 255, 255), -1)

	cv2.imwrite('demo/images/{}/{}.jpg'.format(video_name,i),frame)  
	i += 1

