#include <SoftwareSerial.h>
#include <Servo.h> 
#include "U8glib.h" //OLED 사용을 위한 라이브러리

U8GLIB_SSD1306_128X64 u8g(U8G_I2C_OPT_NONE);  // I2C

int servoPin_1 = 8; //서보모터 3번 핀 지정
int servoPin_2 = 9; //서보모터 3번 핀 지정
int servoPin_3 = 10; //서보모터 3번 핀 지정

int angle = 90; //서보모터가 움직일 각도 변수
int signalNum = 0;

Servo servo;

void setup() {
  Serial.begin(9600);
}

void loop() { 

  /*
  u8g.firstPage();
  do{
    showBasicOLED();
  } while( u8g.nextPage() );*/

  if (Serial.available() > 0){
    
    signalNum = Serial.parseInt();
    Serial.println(signalNum);

     u8g.firstPage();
     do{
      showBasicOLED();
      } while( u8g.nextPage() );


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

//OLED 디스플레이 출력
void showBasicOLED(){

  if (signalNum == -1){

    u8g.setFont(u8g_font_8x13r);

    u8g.setPrintPos(25, 10);
    u8g.print("Please wear");

    u8g.setScale2x2();
    u8g.setPrintPos(0, 18);
    u8g.print("THE MASK");
    u8g.undoScale();

    u8g.setPrintPos(35, 55);
    u8g.print("PROPERLY");
    
  } else {
    
    u8g.setFont(u8g_font_unifont); //폰트 설정
    u8g.setPrintPos(0, 15); //좌표 이동
    u8g.print("Input Number: ");
    u8g.print(signalNum); //문자 외에 변수도 삽입 가능함

    u8g.setFont(u8g_font_8x13r);
    u8g.setPrintPos(0, 22);
    
    u8g.setScale2x2();
    u8g.print("MOTOR");
    
    u8g.setPrintPos(45, 22);
    u8g.print(signalNum);

    u8g.setPrintPos(55, 22);
    u8g.print("R");
    
    u8g.undoScale();

    u8g.setFont(u8g_font_unifont);
    u8g.setPrintPos(0, 62);
    u8g.print("Safe Pass System");
  
  }
  
}
