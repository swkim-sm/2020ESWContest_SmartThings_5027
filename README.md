![KakaoTalk_20200927_135048782](https://user-images.githubusercontent.com/67955977/95685027-550fe280-0c30-11eb-9c4b-4ee9dee266fa.png)
## Team
* 팀이름 : 넌컨택트(Noncontact)
* 팀원 : 김민지(Minji Kim), 김서원(Seowon Kim), 박동연(Dongyoen Park), 이하람(Haram Lee)

## Project
* 목적 : 포스트 코로나 시대를 맞아 공장 내 보다 안전한 작업 환경 조성을 위하여 공장 내 기기 사용 시 작업자의 마스크 착용 여부를 확인하고, 모션 인식으로 기기를 제어하는 시스템
* 목표
    * 근무자의 안전 보호구 착용 여부 확인 및 착용 독려
    * 근무자와 작업 환경 간의 접촉 최소화를 위한 모션 인식 제어
* 시연동영상 https://youtu.be/1E9v1babfgc

## System architecture
![시스템 아키텍쳐](https://user-images.githubusercontent.com/35680202/95684378-47585e00-0c2c-11eb-9444-0d6626f090c2.png)

## How To Use SafePass?
> Virtual Keyboard : 원하는 버튼 위에서 주먹을 쥐었다 펴면 클릭 이벤트가 발생합니다.

![Virtual Keyboard](https://user-images.githubusercontent.com/35680202/95684763-a1f2b980-0c2e-11eb-9c87-64330c50d676.gif)

> 위와 같이 **FAN**과 **ON**을 선택하면 액츄에이터에 의해 스위치가 눌려서 작동됩니다.

![FAN](https://user-images.githubusercontent.com/35680202/95684625-e16cd600-0c2d-11eb-905f-e3461ed7265a.png)


## Prerequisite
### 라즈베리파이 개발 환경 셋팅
1. 아나콘다 프롬프트 or CMD 실행
2. (pip 패키지 업그레이드)
    ```bash
    conda upgrade pip
    pip install upgrade
    ```
3. 가상환경 새로 설치
    ```bash
    conda create -n (env) python=3.7 activate (env)
    ```
4. tensorflow 설치
    ```bash
    pip install tensorflow==1.15.2
    ```
5. 버전 확인
    ```bash
    python import tensorflow as tf tf.__version__
    ```
6. 라이브러리 설치
    ```bash
    pip install numpy matplotlib pillow opencv-python
    pip install opencv-contrib-python
    pip install flask
    pip install werkzeug
    ```
7. 모션인식 모델 다운로드
    * https://github.com/cansik/yolo-hand-detection 접속
    * https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny.cfg 다운로드
    * https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny.weights 다운로드
    * ~/safe_pass/safe_pass/models/ 안에 넣기

### 사용된 도구
* [Tensorflow](https://www.tensorflow.org/api_docs)
* [Raspberry Pi](https://www.raspberrypi.org/documentation/)
* [Arduino](https://www.arduino.cc/reference/en/)
