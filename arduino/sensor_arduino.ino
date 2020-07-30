
int ledBtn = 5; // LED Button(switch) 포트 번호 5
int propellerBtn = 6; // Propeller Button(switch) 포트 번호 6
int piezoBtn = 7; // Piezo Button(switch) 포트 번호 7

int led = 8; // LED 포트 번호 8
int propeller = 9; // Propeller 포트 번호 9
int piezo = 10; // Piezo 포트 번호 10

int actuator_state[3] = {0, 0, 0}; // 액츄에이터 상태 저장 (led, propeller, piezo)
int button_flag[3] = {0, 0, 0}; // 버튼 입력 변화 상황 값 저장 (led, propeller, piezo)
int change_flag[3] = {0, 0, 0}; // 이전 상황 값 저장 (led, propeller, piezo)

void setup() {
  pinMode(led, OUTPUT);
  pinMode(propeller, OUTPUT);
  pinMode(piezo, OUTPUT);

  pinMode(ledBtn, INPUT); //기본이 풀업저항 (누를때 0)
  pinMode(propellerBtn, INPUT); //기본이 풀업저항 (누를때 0)
  pinMode(piezoBtn, INPUT); //기본이 풀업저항 (누를때 0)

  Serial.begin(9600);
}

void loop() {

  button_flag[0] = digitalRead(ledBtn); // led 버튼 상태 읽고 저장
  button_flag[1] = digitalRead(propellerBtn); // propeller 버튼 상태 읽고 저장
  button_flag[2] = digitalRead(piezoBtn); // piezo 버튼 상태 읽고 저장

  //Serial.println(button_flag[0], button_flag[1], button_flag[2]);

  for (int i=0; i<3; i++){
    if((button_flag[i] == HIGH) && (change_flag[i] == LOW)){ //버튼 입력이 있을 때
      actuator_state[i] = 1 - actuator_state[i]; //풀업저항이므로, 누를때(0) led_state = 1 되도록 함
      delay(10); //바운싱 현상 방지를 위한 딜레이
    }
    change_flag[i] = button_flag[i]; //이전 값 저장
  }

  checkLed();
  checkPropeller();
  checkPiezo();

}

void checkLed(){
  if(actuator_state[0] == 1){
    digitalWrite(led, HIGH); //led_state가 1일때 LED 켜기
  }else{
    digitalWrite(led, LOW); //led_state가 0일때 LED 끄기
  }
}

void checkPropeller(){
  if(actuator_state[1] == 1){
    digitalWrite(propeller, HIGH); //propeller_state가 1일때 propeller 동작
  }else{
    digitalWrite(propeller, LOW); //propeller_state가 0일때 propeller 멈춤
  }
}

void checkPiezo(){
  if(actuator_state[2] == 1){
    playAlarm(); //piezo_state가 1일때 LED 소리 알람
  }else{
    noTone(piezo); //piezo_state가 0일때 LED 소리 끄기
  }
}

void playAlarm(){
  tone(piezo, 440);
  delay(200);
  tone(piezo, 440);
  delay(50);
  tone(piezo, 440);
  delay(200);
  tone(piezo, 440);
  delay(50);
}
