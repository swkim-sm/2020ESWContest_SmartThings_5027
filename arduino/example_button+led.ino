int LED = 7; //led 포트 번호 7
int BUTTON = 8; //button(switch) 포트 번호 8

int button_flag = 0; //버튼 입력 변화 상황 값 저장
int change_flag = 0; //이전 상황 값 저장

int led_state = 0; //led 점등 여부 상황 값 저장

void setup() {
  pinMode(LED, OUTPUT);
  pinMode(BUTTON, INPUT); //기본이 풀업저항 (누를때 0)
}

void loop() {

  button_flag = digitalRead(BUTTON); //버튼 상태 저장

  if((button_flag == HIGH) && (change_flag == LOW)){ //버튼 입력이 있을 때
    led_state = 1 - led_state; //풀업저항이므로, 누를때(0) led_state = 1 되도록 함
    delay(10); //바운싱 현상 방지를 위한 딜레이
  }

  change_flag = button_flag; // 이전 값 저장

  if(led_state == 1){
    digitalWrite(LED, HIGH); //led_state가 1일때 LED 켜기
  }else{
    digitalWrite(LED, LOW); //led_state가 0일때 LED 끄기
  }

}
