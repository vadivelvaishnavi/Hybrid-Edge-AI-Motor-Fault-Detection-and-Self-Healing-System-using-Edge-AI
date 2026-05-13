#include "config.h"
#include "one_class_svm_model.h"
#include "autoencoder_model.h"

void setup() {
  Serial.begin(115200);
}

void loop() {

  float rpm = 1800;
  float current = 3.5;
  float vibration = 0.15;
  float temperature = 40;

  Serial.print("RPM: ");
  Serial.println(rpm);

  delay(1000);
}
