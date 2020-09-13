# Safe Pass
## Team
* 팀이름 : 넌컨택트(Noncontact)
* 팀원 : 김민지(Minji Kim), 김서원(Seowon Kim), 박동연(Dongyoen Park), 이하람(Haram Lee)

## Project
* 목적 : 포스트 코로나 시대를 맞아 공장 내 보다 안전한 작업 환경 조성을 위하여 공장 내 기기 사용 시 작업자의 마스크 착용 여부를 확인하고, 모션 인식으로 기기를 제어하는 시스템
* 목표
    * 근무자의 안전 보호구 착용 여부 확인 및 착용 독려
    * 근무자와 작업 환경 간의 접촉 최소화를 위한 모션 인식 제어

## 설치하기
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
    ```

### 서버 개발 환경 셋팅

## 테스트 실행하기
이 시스템을 위한 자동화된 테스트를 실행하는 방법을 적어주세요.

## 배포
추가로 실제 시스템에 배포하는 방법을 노트해 두세요.

## 사용된 도구
* [Tensorflow](https://www.tensorflow.org/api_docs)
* [Raspberry Pi](https://www.raspberrypi.org/documentation/)
* [Arduino](https://www.arduino.cc/reference/en/)
