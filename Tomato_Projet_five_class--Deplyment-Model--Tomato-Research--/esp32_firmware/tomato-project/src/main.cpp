/*
 * Tomato sorter -- Bluetooth JSON control
 * ----------------------------------------
 * Receives {"command": ..., "value": ...} over Bluetooth SPP (BTConnecter)
 * and drives the hardware below. IR sensor changes are pushed back to the
 * laptop the same way, unprompted.
 *
 * ---- WIRING ----
 * DM542 stepper driver (belt motor):
 *   PUL+ -> ESP32 GPIO 25   PUL- -> GND
 *   DIR+ -> ESP32 GPIO 26   DIR- -> GND
 *   (ENA not used -- driver runs enabled by default)
 *
 * 4x servo (sorting gates), signal wire only -- power from a separate 5-6V
 * supply with GND tied to the ESP32's GND, NOT from the ESP32 5V pin:
 *   servo1 -> GPIO 12   servo2 -> GPIO 13
 *   servo3 -> GPIO 14   servo4 -> GPIO 27
 *
 * 4x IR sensor (digital out):
 *   IR1 -> GPIO 18   IR2 -> GPIO 19
 *   IR3 -> GPIO 21   IR4 -> GPIO 23
 *
 * ---- COMMAND PROTOCOL (matches DataModel's allowed commands) ----
 *   "servo1".."servo4"  value = target angle in degrees (0-180)
 *   "motor"             value = signed speed in steps/sec
 *                        (sign = direction, 0 = stop; belt runs continuously
 *                        at that speed until the next "motor" command)
 *   "IR1".."IR4"        outbound only -- sent to the laptop whenever that
 *                        sensor's reading changes, never accepted inbound
 */

#include <Arduino.h>
#include <ESP32Servo.h>
#include "BTConnecter.h"
#include "DataModel.h"

BTConnecter btConnecter("TomatoSorter");

void handleIncoming();
void applyCommand(const DataModel &model);
void setMotorSpeed(float stepsPerSecond);
void runStepper();
void pollIRSensors();

// ---------------- Servos (gates) ----------------
const int SERVO_PINS[4] = {13, 12, 14, 27};
const char *SERVO_COMMANDS[4] = {"servo1", "servo2", "servo3", "servo4"};
Servo gateServos[4];

// ---------------- Stepper (belt, DM542) ----------------
const int STEP_PIN = 25; // -> PUL+
const int DIR_PIN = 26;  // -> DIR+

unsigned long stepIntervalUs = 0; // 0 = stopped, else time between pin toggles
unsigned long lastStepTime = 0;
bool stepPinState = false;

// ---------------- IR sensors ----------------
const int IR_PINS[4] = {18, 19, 21, 23};
const char *IR_COMMANDS[4] = {"IR1", "IR2", "IR3", "IR4"};
int lastIRState[4] = {-1, -1, -1, -1};    // last state actually reported
int pendingIRState[4] = {-1, -1, -1, -1}; // most recent raw reading, awaiting debounce
unsigned long pendingIRSince[4] = {0, 0, 0, 0};
const unsigned long IR_DEBOUNCE_MS = 20; // reading must hold steady this long before it's reported

void setup() {
  // Physical (USB) serial for now, instead of Bluetooth -- same JSON
  // protocol either way. Debug prints below share this same port, so the
  // PC-side reader just needs to ignore any non-JSON lines it sees.
  btConnecter.beginSerial(Serial, 115200);
  Serial.println("BTConnecter ready over USB serial.");

  for (int i = 0; i < 4; i++) {
    gateServos[i].setPeriodHertz(50);
    gateServos[i].attach(SERVO_PINS[i], 500, 2400);
  }

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  // INPUT_PULLUP suits open-collector IR modules; harmless for push-pull
  // modules too since they'll just drive over the weak pull-up.
  for (int i = 0; i < 4; i++) {
    pinMode(IR_PINS[i], INPUT_PULLUP);
  }
}

void loop() {
  handleIncoming();
  runStepper();
  pollIRSensors();
}

// Reads one JSON command from Bluetooth (if any arrived) and applies it.
void handleIncoming() {
  JsonDocument doc;
  if (!btConnecter.receiveSignal(doc)) {
    return;
  }

  DataModel model;
  if (!DataModel::fromJson(doc, model)) {
    Serial.println("[BT] received JSON with an unrecognized/missing command, ignored");
    return;
  }

  applyCommand(model);
}

void applyCommand(const DataModel &model) {
  String command = model.getCommand();
  float value = model.getValue();

  for (int i = 0; i < 4; i++) {
    if (command == SERVO_COMMANDS[i]) {
      float angle = constrain(value, 0.0f, 180.0f);
      gateServos[i].write((int)angle);
      Serial.printf("[CMD] %s -> %.1f deg\n", SERVO_COMMANDS[i], angle);
      return;
    }
  }

  if (command == "motor") {
    setMotorSpeed(value);
    Serial.printf("[CMD] motor -> %.1f steps/sec\n", value);
    return;
  }

  // "IR1".."IR4" are sensor readings the ESP32 reports out, not commands
  // the PC should send in -- nothing to do if one shows up inbound.
}

// value's sign sets belt direction, magnitude sets speed. 0 stops the belt.
void setMotorSpeed(float stepsPerSecond) {
  if (fabs(stepsPerSecond) < 0.01f) {
    stepIntervalUs = 0;
    return;
  }

  digitalWrite(DIR_PIN, stepsPerSecond >= 0 ? HIGH : LOW);
  // Two pin toggles per step, so the toggle rate is 2x the step rate.
  // Clamp to at least 1us -- at very high requested speeds this would
  // otherwise round down to 0, which runStepper() treats as "stopped".
  unsigned long interval = (unsigned long)(1000000.0f / (2.0f * fabs(stepsPerSecond)));
  stepIntervalUs = interval == 0 ? 1 : interval;
}

// Non-blocking step pulse generator -- runs every loop() regardless of
// Bluetooth traffic, same pattern as the gate/IR handling.
void runStepper() {
  if (stepIntervalUs == 0) {
    return;
  }
  unsigned long now = micros();
  if (now - lastStepTime >= stepIntervalUs) {
    lastStepTime = now;
    stepPinState = !stepPinState;
    digitalWrite(STEP_PIN, stepPinState ? HIGH : LOW);
  }
}

// Sends an IR* JSON message to the laptop only when that sensor's reading
// actually changes AND has held steady for IR_DEBOUNCE_MS. Without this,
// a floating/unconnected IR pin can flip rapidly from electrical noise and
// flood Serial + Bluetooth with prints -- that flooding was slowing loop()
// down enough to disrupt the stepper's pulse timing (servos, which only
// react to occasional incoming commands, weren't affected the same way).
void pollIRSensors() {
  unsigned long now = millis();
  for (int i = 0; i < 4; i++) {
    int reading = digitalRead(IR_PINS[i]);

    if (reading != pendingIRState[i]) {
      pendingIRState[i] = reading;
      pendingIRSince[i] = now;
      continue; // just changed -- wait to see if it's real or noise
    }

    if (reading == lastIRState[i]) {
      continue; // no change from what was last reported
    }

    if (now - pendingIRSince[i] < IR_DEBOUNCE_MS) {
      continue; // hasn't been stable long enough yet
    }

    lastIRState[i] = reading;

    JsonDocument out;
    out["command"] = IR_COMMANDS[i];
    out["value"] = (float)reading;
    btConnecter.sendSignal(out);

    Serial.printf("[IR] %s -> %d\n", IR_COMMANDS[i], reading);
  }
}