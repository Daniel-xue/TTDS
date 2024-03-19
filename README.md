# TTDS

## Introduction
球類運動的自動化追蹤與分析為近年許多研究所探討的問題。先前方法多數仰賴高規格且多台固定攝影角度的攝影設備，成本高昂。
隨著機器學習在影像辨識以及物件偵測的技術逐漸成熟，期望設計新的系統，僅需要單個固定相機所錄下的影像即可判斷球體移動的軌跡與落點，進而降低自動化追蹤導入運動競技賽事的成本，這將促進科學和運動競技的結合。

## system architecture
![image](https://github.com/Daniel-xue/TTDS/blob/main/system_architecture.PNG)

系統主要由三個模塊組成: 球偵測、球桌偵測、落點偵測。

## 球偵測
![image](https://github.com/Daniel-xue/TTDS/blob/main/%E7%90%83%E5%81%B5%E6%B8%AC.PNG)
![image](https://github.com/Daniel-xue/TTDS/blob/main/%E7%90%83%E5%81%B5%E6%B8%AC2.PNG)

對於桌球的偵測，採用TrackNet-R模型做訓練和預測。

## 球桌偵測
![image](https://github.com/Daniel-xue/TTDS/blob/main/%E7%90%83%E6%A1%8C%E5%81%B5%E6%B8%AC.PNG)

球桌的偵測是運用了傳統的影像處理技術，透過四邊形的白邊特徵找出球桌。

## 落點偵測
![image](https://github.com/Daniel-xue/TTDS/blob/main/%E8%90%BD%E9%BB%9E%E5%81%B5%E6%B8%AC.PNG)

首先利用球偵測預測出球座標資訊(x和y)，再對前後兩格影像的球座標(x1,y1)和(x2,y2)計算歐機里德距離，接著除上FPS得到速度v。落點的偵測運用了x、y、v的資訊作為特徵值，並以當前點為中心，加上前後各三點，共7點組成一筆資料作為輸入，這麼做是為了利用連續的訊息更好的找出落點，而輸出則是以0和1標示是否為落點，模型採用的是6層全連接層的MLP，訓練了600epochs，batch_size為16，最後得到的效能"TPR是84%"，"FPR是20%"。

## 參考
[1] https://arxiv.org/abs/1907.03698 TrackNet：用於追蹤運動應用中高速和微小物體的深度學習網絡

[2] https://arxiv.org/abs/1708.02002 密集物體偵測的焦點損失

[3] https://github.com/ChgygLin/TrackNetV2-pytorch TrackNetV2：高效率 TrackNet (GitLab) 

[4] https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2 


