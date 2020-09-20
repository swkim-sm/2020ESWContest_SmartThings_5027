#include <SoftwareSerial.h>
#include <Servo.h> 

int servoPin_LED = 8; //서보모터 8번 핀 지정 - LED 제어
int servoPin_FAN = 9; //서보모터 9번 핀 지정 - 환풍기 제어
int servoPin_CB = 10; //서보모터10번 핀 지정 - 컨베이어 벨트 제어

int angle = 90;
int signalNum = 0;

Servo servo;

void changeAngle(int startAngle, int endAngle, int motorNum);

void setup() {
  Serial.begin(115200);
}

void loop() {

  int signalNum = 0;

  if (Serial.available() > 0){
    char ch = Serial.read();
    signalNum = ch -'0';
    //signalNum = Serial.parseInt();
    Serial.println(signalNum);

    switch (signalNum) {
      case 1: //창고 LED ON
      Serial.println(signalNum);
        changeAngle(0,30, servoPin_LED);
        changeAngle(30,0, servoPin_LED);
        break;

      case 2: //창고 LED OFF5
        changeAngle(0,30, servoPin_LED);
        changeAngle(30,0, servoPin_LED);
        break;

      case 3: //환풍기 ON
        changeAngle(0,30, servoPin_FAN);
        changeAngle(30,0, servoPin_FAN);
        break;
      case 4: //환풍기 OFF
        changeAngle(0,30, servoPin_FAN);
        changeAngle(30,0, servoPin_FAN);
        break;
      case 5: //컨베이어 벨트 ON
        changeAngle(0, 100, servoPin_CB);
        break;
      case 6: //컨베이어 벨트 OFF
        changeAngle(90, 0, servoPin_CB);
        break;
    }
  }
}

void changeAngle(int startAngle, int endAngle, int motorNum){

  //서보모터 객체에 움직일 모터 정보 전달
  servo.attach(motorNum);
  //Serial.println(motorNum);

  if(startAngle < endAngle){
    for(angle = startAngle; angle < endAngle; angle++) {
      servo.write(angle);
      delay(15);
    }
  } else { // startAnlge > endAngle
    for(angle = startAngle; angle > endAngle; angle--) {
      servo.write(angle);
      delay(15);
    }
  }

}