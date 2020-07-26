#include <SoftwareSerial.h>
#include <Servo.h> 

int servoPin = 3; //서보모터 3번 핀 지정

Servo servo; //서보모터 객체 생성

int angle = 90; //서보모터가 움직일 각도 변수
int signalNum = -1;

void setup() { 
    servo.attach(servoPin); // 서보모터 핀 입력
    Serial.begin(9600);
}

void loop() { 

  if (Serial.available() > 0){
    
    signalNum = Serial.parseInt();
    Serial.println(signalNum);

    if (signalNum == 0){
      changeAngle(90, 40);
      changeAngle(40, 90);
    } else if (signalNum == 1){
      changeAngle(90, 140);
      changeAngle(140, 90);
    } else{
      angle = 90;
      servo.write(angle); 
    }
 
  }
  
}

void changeAngle(int startAngle, int endAngle){

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
