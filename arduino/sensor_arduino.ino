
int ledBtn = 6; // LED Button(switch) 포트 번호 5 - 버튼 스위치
int fanBtn = 7; // Propeller Button(switch) 포트 번호 6 - 토글 스위치

int led = 8; // LED 포트 번호 8
int fan = 9; // Propeller 포트 번호 9

int machine_state[2] = {0, 0}; // 액츄에이터 상태 저장 (led, fan)
int button_flag[2] = {0, 0}; // 버튼 입력 변화 상황 값 저장 (led, fan)
int change_flag[2] = {0, 0}; // 이전 상황 값 저장 (led, fan)

void setup() {
  pinMode(led, OUTPUT);
  pinMode(fan, OUTPUT);

  pinMode(ledBtn, INPUT_PULLUP); //기본이 풀업저항 (누를때 0) - 버튼 스위치
  pinMode(fanBtn, INPUT_PULLUP); //기본이 풀업저항 (누를때 0) - 토글 스위치

  Serial.begin(9600);
}

void loop() {

  button_flag[0] = digitalRead(ledBtn); // led 버튼 상태 읽고 저장
  button_flag[1] = digitalRead(fanBtn); // propeller 버튼 상태 읽고 저장

  Serial.print(button_flag/ a 1/[0]);
  Serial.println(button_flag[1]);

  for (int i=0; i<2; i++){
    if((button_flag[i] == HIGH) && (change_flag[i] == LOW)){ //버튼 입력이 있을 때
      machine_state[i] = 1 - machine_state[i]; //풀업저항이므로, 누를때(0) led_state = 1 되도록 함
      delay(10); //바운싱 현상 방지를 위한 딜레이
    }
    change_flag[i] = button_flag[i]; //이전 값 저장
  }

  checkLed();
  checkFan();

}

void checkLed(){
  if(machine_state[0] == 1){
    digitalWrite(led, HIGH); //led_state가 1일때 LED 켜기
  }else{
    digitalWrite(led, LOW); //led_state가 0일때 LED 끄기
  }
}

void checkFan(){
  if(machine_state[1] == 1){
    digitalWrite(fan, HIGH); //propeller_state가 1일때 propeller 동작
  }else{
    digitalWrite(fan, LOW); //propeller_state가 0일때 propeller 멈춤
  }
}