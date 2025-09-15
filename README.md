# TTDS

## Introduction
桌球運動的自動化追蹤與分析為近年許多研究所探討的問題。傳統方法多數仰賴高規格且多台固定攝影角度的攝影設備，成本高昂。
期望設計一個單目視覺系統，僅需依靠單個固定相機所錄下的影像即可判斷球體移動的軌跡與落點，進而降低自動化追蹤導入運動競技賽事的成本，這將促進科學和運動競技的結合。
提出的架構包含使用深度學習網路的桌球辨識，還有基於傳統影像處理技術實現的球桌偵測，最後結合分類模型來預測是否為落點。

## system architecture
![image](https://github.com/Daniel-xue/TTDS/blob/main/system_architecture.PNG)

系統主要由三個模塊組成: 球偵測、球桌偵測、落點偵測。

## 球偵測
![image](https://github.com/Daniel-xue/TTDS/blob/main/%E7%90%83%E5%81%B5%E6%B8%AC.PNG)
![image](https://github.com/Daniel-xue/TTDS/blob/main/%E7%90%83%E5%81%B5%E6%B8%AC2.PNG)

採用TrackNet-R模型。
TrackNet 是由台灣國立交通大學發明的用於高速微小物體追蹤的深度學習網路。它是一個 FCN 模型，採用 VGG16 產生特徵圖，並使用 DeconvNet 進行像素級分類解碼。
TrackNet 可以將多個連續幀(3張)作為輸入，模型不僅可以學習物體追踪，還可以學習軌跡，從而增強其定位和識別能力。
TrackNet 會產生以球為中心的高斯熱圖來指示球的位置。二值交叉熵用作損失函數，用於計算預測熱圖與真實熱圖之間的差異。
此系統不僅能考量球體運動慣性，並能夠有效解決影像背景較複雜亦或是球體被物件遮蔽的狀況。
使用了幀率40 fps的桌球比賽錄像訓練(總計4萬多張照片)，球偵測"Precision"是86.9%，"Recall"是83.4%。

## 球桌偵測
![image](https://github.com/Daniel-xue/TTDS/blob/main/%E7%90%83%E6%A1%8C%E5%81%B5%E6%B8%AC.PNG)

本專案運用了多種影像處理技術（GrayScale、HoughLinesP、MergeLines、FindHomography、PerspectiveTransform），  
透過球桌邊框的四邊形白邊特徵進行分析與辨識，成功實現球桌位置的自動偵測。。

## 落點偵測
![image](https://github.com/Daniel-xue/TTDS/blob/main/%E8%90%BD%E9%BB%9E%E5%81%B5%E6%B8%AC.PNG)

模型使用有6層全連接層的MLP，訓練600輪，批次大小為16。
在得到影像中每一幀的球座標後，為屬於落點的畫面標記1，不屬於落點的標記0，並使用連續幀(7張)來訓練模型。
在我們的測試集中(大約4千多張照片)，落點分類"TPR"是84%，"FPR"是20%。

## 參考
[1] https://arxiv.org/abs/1907.03698 TrackNet：用於追蹤運動應用中高速和微小物體的深度學習網絡

[2] https://arxiv.org/abs/1708.02002 密集物體偵測的焦點損失

[3] https://github.com/ChgygLin/TrackNetV2-pytorch TrackNetV2：高效率 TrackNet (GitLab) 

[4] https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2 TrackNet-羽球-追蹤-tensorflow2


