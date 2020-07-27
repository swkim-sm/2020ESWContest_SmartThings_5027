#include <SoftwareSerial.h>
#include <Servo.h> 

int servoPin_1 = 8; //서보모터 3번 핀 지정
int servoPin_2 = 9; //서보모터 3번 핀 지정
int servoPin_3 = 10; //서보모터 3번 핀 지정

int angle = 90; //서보모터가 움직일 각도 변수
int signalNum = -1;

Servo servo;

void setup() {
  Serial.begin(9600);
}

void loop() { 

  if (Serial.available() > 0){
    
    signalNum = Serial.parseInt();
    Serial.println(signalNum);

    if (signalNum == 1){
      changeAngle(90, 40, signalNum);
      changeAngle(40, 140, signalNum);
      changeAngle(140, 90, signalNum);
    } else if (signalNum == 2){
      changeAngle(90, 40, signalNum);
      changeAngle(40, 140, signalNum);
      changeAngle(140, 90, signalNum);
    } else if (signalNum == 3){
      changeAngle(90, 40, signalNum);
      changeAngle(40, 140, signalNum);
      changeAngle(140, 90, signalNum);
    } else{
      angle = 90;
    }
 
  }

  signalNum = 0;
  
}

void changeAngle(int startAngle, int endAngle, int motorNum){
  
  int servo_pin = 0;

  if (motorNum==1){
    servo_pin = servoPin_1;
  } else if (motorNum==2){
    servo_pin = servoPin_2;
  } else if (motorNum==3){
    servo_pin = servoPin_3;
  } else{
    Serial.println("MotorNum Error");
  }

  servo.attach(servo_pin);

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
