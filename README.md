# Project 집PT
프로젝트 정리 Notion 링크 : https://www.notion.so/Intel-Edge-AI-d020b3a31ccc44c1b93422d7db8c7161

![image](https://github.com/user-attachments/assets/9866e84e-500d-4de0-818c-f49d1235d711)

* 프로젝트 배경

   - 최근 들어 e러닝과 비대면 수업이 급격히 증가하면서 학생들의 집중력이 저하되는 문제가 발생하고 있습니다. 이러한 문제는 온라인 수업 환경에서 특히 두드러지며, 학생들이 졸거나 하품을 하거나 자세가 나빠지는 등 다양한 형태로 나타나고 있습니다. 이러한 문제를 해결하기 위해, 학생들의 수업 태도를 실시간으로 모니터링하고 분석하는 시스템이 필요합니다.

* 프로젝트 목적

   - 이 프로젝트의 목적은 딥러닝을 활용해 학생들의 수업 태도를 실시간으로 모니터링하여 집중력, 졸음, 자세 등의 상태를 파악하고, 이를 통해 학생들의 학습 집중도를 향상시키며 올바른 학습 자세를 유도하는 것입니다. 이를 통해 학습 효율성을 높이고 건강 문제를 예방할 수 있습니다.

### 팀원 역할 표

| 성명   | 역할                                                      |
|--------|-----------------------------------------------------------|
| 김승민 | 자세 추정 데이터 수집/전처리, 자세 추정 모델 학습          |
| 신나라 | 자세 추정 데이터 수집/전처리, 자세 추정 모델 학습          |
| 임지원 | 학습 태도 기능 구현                                       |
| 이지원 | 학습 태도 기능 구현                                       |
| 조원진 | 학습 태도 기능 구현, 웹 개발                              |
| 조수환 | 프로젝트 총괄, 웹 개발                                     |

## High Level Design

### 프로젝트 아키텍처

#### 시스템 개요
* 이 시스템은 학생들의 수업 태도를 모니터링하기 위해 웹캠을 이용하여 학생들의 태도와 자세를 분석하는 시스템입니다. 이를 위해 딥러닝 모델을 사용하여 실시간으로 데이터를 처리하고, 웹 애플리케이션의 대시보드를 통해 학생과 교사에게 피드백을 제공합니다.

#### 시스템 구조
#### 프론트엔드
* 웹 애플리케이션
    * 로그인 페이지
    * 웹캠 출력 페이지
    * 대시보드 페이지 (상태 모니터링 결과 표시)
#### 백엔드
* API 서버
    * 웹캠 이미지 데이터 수신
    * 데이터 처리 및 분석
    분석 결과 저장 및 제공
#### 데이터베이스
* 데이터 저장
    * 학생 정보
    * 모니터링 로그 (눈 개폐, 하품 횟수, 자세 등)
#### 기술 스택
* 프로그래밍 언어: Python
* 프레임워크: Django(백엔드), HTML,CSS,JS(프론트엔드)
* 라이브러리:  MediaPipe(안면 인식), OpenCV(영상 처리), yolov8-pose(자세 분석)
* 데이터베이스: SQLite

### 데이터 흐름도
![image](https://github.com/user-attachments/assets/6f903d7f-cb58-4301-bafa-fadc712e3086)


### 기능 요구사항
1. 하품 횟수 파악
   * 웹캠을 통해 실시간으로 하품 횟수를 감지하고 기록합니다.
3. 눈 개폐 여부 파악
   * 눈의 개폐 여부를 실시간으로 모니터링하여 졸음 상태를 판단합니다.
3. 자세 파악
   * 학생의 자세를 분석하여 고개 떨굼, 구부정한 자세 등을 실시간으로 감지합니다.
5. 대시보드
   * 학생의 상태를 실시간으로 시각화하여 보여줍니다.

### 구현 방법
1. 데이터 수집 및 전처리
   * 웹캠과 yolov8-pose 모델을 통해 학생의 자세 데이터를 수집했습니다. 수집된 데이터는 주요 포즈 키포인트(관절 위치)를 포함하며, 자세 인식을 위한 주요 특징으로 사용되었습니다.

2. 모델 학습
   * 포즈 키포인트 데이터를 활용하여 자세 추정 모델을 학습했습니다. 이를 위해 yolov8-pose 모델로 추출한 키포인트 데이터를 기반으로 자체 개발한 머신러닝 분류 모델을 사용하여 다양한 자세(예: 고개 떨굼, 구부정한 자세 등)를 정확하게 인식할 수 있도록 훈련되었습니다.

3. 기능 구현
   * 자세 추정: 자체 학습한 머신러닝 분류 모델을 통해 실시간으로 학생의 자세를 분석하고, 올바르지 않은 자세(예: 고개 떨굼, 구부정한 자세 등)를 감지합니다.
   * 눈 개폐 및 하품 감지: MediaPipe 모델을 사용하여 학생의 눈 개폐 여부와 하품 횟수를 실시간으로 모니터링합니다.
   * 시선 추적: MediaPipe를 통해 학생의 시선을 추적하고, 학생이 화면을 주시하고 있는지 여부를 감지하여 수업 집중도를 평가합니다.

## Clone code
```
git clone https://github.com/suhwanjo/Intel-Edge-AI-SW-Academy-Vision-AI.git
```

## Prerequite
```
cd ~/Intel-Edge-AI-SW-Academy-Vision-AI/django
python -m venv .venv
source .venv/bin/activate
(.venv) pip install -r requirments.txt
```
## Steps to run
```
cd ~/Intel-Edge-AI-SW-Academy-Vision-AI/django
source .venv/bin/activate
(.venv) python manage.py makemigrations
(.venv) python manage.py migrate
(.venv) python manage.py runserver
```

## Output
* 시연 영상 참고
