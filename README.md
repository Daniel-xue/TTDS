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
一種兩階段式的深度學習網路軌跡預測方法。網路的第一階段負責將影像中包含球體的區域進行選擇，第二階段透過前述階段產生的連續畫面(3張)做為輸入，提升球偵測預測效能。
此系統不僅能考量球體運動慣性，並能夠有效解決影像背景較複雜亦或球體被物件所遮蔽的狀況。
使用了幀率40 fps的桌球比賽錄像訓練(總計4萬多張照片)，球偵測"Precision"是86.9%，"Recall"是83.4%。

## 球桌偵測
![image](https://github.com/Daniel-xue/TTDS/blob/main/%E7%90%83%E6%A1%8C%E5%81%B5%E6%B8%AC.PNG)

運用了傳統的影像處理技術，透過球桌的四邊形白邊特徵去偵測。

## 落點偵測
![image](https://github.com/Daniel-xue/TTDS/blob/main/%E8%90%BD%E9%BB%9E%E5%81%B5%E6%B8%AC.PNG)

使用了有6層全連接層的MLP。
在有了所有畫面中球座標後，為屬於落點的畫面做標記，並使用連續畫面(7張)中球的座標變化來訓練模型。
的影片中，落點分類"TPR"是84%，"FPR"是20%。

## 參考
[1] https://arxiv.org/abs/1907.03698 TrackNet：用於追蹤運動應用中高速和微小物體的深度學習網絡

[2] https://arxiv.org/abs/1708.02002 密集物體偵測的焦點損失

[3] https://github.com/ChgygLin/TrackNetV2-pytorch TrackNetV2：高效率 TrackNet (GitLab) 

[4] https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2 


