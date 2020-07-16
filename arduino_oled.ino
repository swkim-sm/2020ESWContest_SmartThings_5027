//OLED 제어 코드

#include "U8glib.h" //OLED 라이브러리가 있으나 오류가 있는 경우가 많으므로 대부분 해당 라이브러리 사용

U8GLIB_SSD1306_128X64 u8g(U8G_I2C_OPT_NONE);  // I2C

void setup() {
  //OLED는 이미 핀세팅이 UNO 기준으로 예약되어있으므로 변경이 불가능합니다.
  //GND - Ground
  //VCC - 5V
  //SDA(Serial Data Line) - A4 & SDA
  //SCL(Serial Clock) - A5 & SCL
}

void draw() { // 디스플레이에 계속 출력할 내용을 적은 함수

  u8g.setFont(u8g_font_unifont); //폰트 설정
  u8g.setPrintPos(0, 15); //좌표 이동
  u8g.print("Temp "); //문자 출력 - drawStr(좌,표, 내용) 으로 동시에 작성도 가능
  u8g.print("24"); //문자 외에 변수도 삽입 가능함
  u8g.print(" C");

  u8g.setPrintPos(56, 7);
  u8g.print(".");

  u8g.setPrintPos(88, 15);
  u8g.print("H ");
  u8g.print("30");
  u8g.print("%");

  u8g.setFont(u8g_font_8x13r);
  u8g.setPrintPos(0, 22);
  u8g.setScale2x2();
  u8g.print("ISODA");
  u8g.undoScale();

  u8g.setFont(u8g_font_unifont);
  u8g.setPrintPos(88, 44);
  u8g.print("LAB");

  u8g.setFont(u8g_font_unifont);
  u8g.setPrintPos(0, 62);
  u8g.print("SOOKMYUNG W UNIV");

}


void loop() {

  u8g.firstPage();
  do {
    draw();
  } while ( u8g.nextPage() );

  delay(100);
}
