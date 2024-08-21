import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#CSV 파일 불러오기
df_jung = pd.read_csv("./jungjasae.csv")
df_gubu = pd.read_csv("./gubu.csv")
df_gogae = pd.read_csv("./gogae.csv")

#결측치 지우기
df_gubu = df_gubu.dropna()
df_gogae = df_gogae.dropna()

#결과값 추가하기      1 : 정자세 , 0 : 구부정한 자세, 2 : 고개숙임
df_jung['result'] = 1
df_gubu['result'] = 0
df_gogae['result'] = 2

#데이터 합치기
total = pd.concat([df_jung, df_gubu, df_gogae], axis=0).reset_index(drop=True)

#데이터 섞기
total = total.sample(frac=1)

#컬럼 초기화
new_index = list(range(1, 1188))
total.index = new_index

#train, test set 분할
train = total.iloc[0:700]
test = total.iloc[700:]

print(total.columns)

X_train = train[['nose_y', 'L_eye_y', 'L_ear_y', 'L_shoulder_y']]
y_train = train['result']

X_test = test[['nose_y', 'L_eye_y', 'L_ear_y', 'L_shoulder_y']]
y_test = test['result']

#모델 학습하기
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)

pred = model_knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred)}")

import pickle

knnPickle = open('model_knn.h5', 'wb')
# source, destination 
pickle.dump(model_knn, knnPickle)
# close the file
knnPickle.close()

import ultralytics

ultralytics.checks()

import cv2
import torch
import pickle
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import accuracy_score

# Load the YOLO model
yolo_model = YOLO('yolov8n-pose.pt')  # Make sure you have the correct path to the YOLOv8 model

# Load the KNN model from disk
with open('model_knn.h5', 'rb') as f:
    model_knn = pickle.load(f)

# 웹캠 캡처 객체 생성
cap = cv2.VideoCapture(0)

frame_count = 0  # 프레임 카운트 초기화

while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # YOLO model로 포즈 추정
    results = yolo_model(frame, conf=0.7)
        
    # 결과 렌더링
    annotated_frame = results[0].plot()
    
    # 포즈 추정 좌표 가져오기
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy[0]  # 첫 번째 사람의 keypoints 가져오기
        print(f"Keypoints detected: {keypoints}")  # Debug print statement
    else:
        keypoints = None
        print("No keypoints detected")  # Debug print statement
    
    if keypoints is not None and keypoints.shape[0] == 17:  # YOLOv8-pose는 보통 17개의 keypoints를 출력
        # Ensure that the necessary keypoints are valid (non-zero)
        if keypoints[0, 1] != 0 and keypoints[2, 1] != 0 and keypoints[4, 1] != 0 and keypoints[5, 1] != 0:
            # 추출한 좌표에서 필요한 y좌표 추출
            nose_y = keypoints[0, 1]   # 코 y좌표
            L_eye_y = keypoints[2, 1]  # 왼쪽 눈 y좌표
            L_ear_y = keypoints[4, 1]  # 왼쪽 귀 y좌표
            R_shoulder_y = keypoints[5, 1]  # 오른쪽 어깨 y좌표

            print(f"nose_y : {nose_y}, L_eye_y: {L_eye_y}, L_ear_y: {L_ear_y}, R_shoulder_y: {R_shoulder_y}")  # Debug print statement
                
            # KNN 모델로 자세 분류
            input_data = np.array([[nose_y, L_eye_y, L_ear_y, R_shoulder_y]])
            pred = model_knn.predict(input_data)
                
            # 결과 출력
            if pred[0] == 1 :
                result_text = "jungjasae"
            elif pred[0] == 0 :
                result_text = "gubujung"
            else :
                result_text = "gogae "
            print(f"Predicted posture: {result_text}")  # Debug print statement
            cv2.putText(annotated_frame, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print("Required keypoints are not detected properly")  # Debug print statement
    else:
        print("Not enough keypoints detected or keypoints are None")  # Debug print statement

    # OpenCV로 렌더링된 프레임 표시
    cv2.imshow("Pose Estimation", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

# 테스트 세트 정확도 출력
# Note: X_test and y_test should be defined in your script where you trained your KNN model
# Ensure X_test and y_test are properly loaded or defined
# pred_test = model_knn.predict(X_test)
# print(f"Accuracy on test set: {accuracy_score(y_test, pred_test)}")