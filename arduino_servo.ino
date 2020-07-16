#include <Servo.h> 

int servoPin = 2; //서보모터 2번 핀 지정

Servo servo; //서보모터 객체 생성

int angle = 0; //서보모터가 움직일 각도 변수 

void setup() { 
    servo.attach(servoPin); // 서보모터 핀 입력
} 

void loop() { 

  // 모터의 뻑뻑한 정도에 따라서 유연하게 회전되지 않을 가능성을 고려하여 90~200도로 설정
  // 추후 하드웨어 설비 상태에 따라서 각도는 유연하게 조정할 예정
  
  // 90도에서 200도로 회전
  for(angle = 90; angle < 200; angle++) { 
    servo.write(angle); 
    delay(15); 
  }
  
  // 200도에서 90도로 회전
  for(angle = 200; angle > 90; angle--) { 
    servo.write(angle); 
    delay(15); 
  }
  
} 
