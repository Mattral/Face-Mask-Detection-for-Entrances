#include <LiquidCrystal.h>
#include <Servo.h>

const int buzzerPin = 13;
int incomingByte;
const int rsPin = 7, enPin = 8, d4Pin = 9, d5Pin = 10, d6Pin = 11, d7Pin = 12;
LiquidCrystal lcd(rsPin, enPin, d4Pin, d5Pin, d6Pin, d7Pin);
Servo doorServo;

void setup() {
  Serial.begin(9600);
  doorServo.attach(6);
  pinMode(buzzerPin, OUTPUT);
  lcd.begin(16, 2);
  Serial.println("Setup");
}

void loop() {
  Serial.println("Loop");

  if (Serial.available() > 0) {
    incomingByte = Serial.read();

    if (incomingByte == 'H') {
      noTone(buzzerPin);
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Mask Detected");

      for (int pos = 0; pos <= 90; pos += 10) {
        doorServo.write(pos);
        delay(100);
        Serial.println("OPEN");
      }
    } else if (incomingByte == 'L') {
      tone(buzzerPin, 450);
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Please Wear Mask");

      for (int pos = 90; pos >= 0; pos -= 10) {
        doorServo.write(pos);
        delay(100);
        Serial.println("Closed");
      }
    } else if (incomingByte == 'N') {
      doorServo.write(0);
    }
  }
}
