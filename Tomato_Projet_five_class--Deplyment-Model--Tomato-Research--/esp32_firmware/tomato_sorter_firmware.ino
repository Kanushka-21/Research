/*
 * Tomato Sorter Firmware (ESP32)
 * ------------------------------
 * Two jobs running at the same time on the same ESP32:
 *   1. Keep the conveyor belt (NEMA23 stepper via DM542 driver) moving.
 *   2. Listen on Serial for one-letter commands from the PC (sent by
 *      conveyor_integration.py) and open the matching sorting gate.
 *
 * Serial protocol (matches conveyor_integration.py, 115200 baud):
 *   'G' = open green gate     'B' = open breaker gate
 *   'T' = open turning gate   'R' = open red gate
 *   (no letter is ever sent for "defect" - it just rides to the end, on purpose)
 *
 * ---- WIRING (from your notes) ----
 * DM542 driver:
 *   PUL+, DIR+, ENA+  -> ESP32 3V3   (tied high - common-anode wiring)
 *   PUL-              -> ESP32 GPIO 18
 *   DIR-              -> ESP32 GPIO 19
 *   GND               -> ESP32 GND
 *   ENA- left unconnected -> driver is always enabled (no code needed for that)
 *
 * 4x MG995 servos (gate order along the belt: green -> breaker -> turning -> red):
 *   All 4 brown (ground) wires tied together -> one wire -> common GND
 *   All 4 red (power) wires tied together    -> one wire -> +5V/+6V from a SEPARATE
 *                                                supply (do NOT power 4 servos from
 *                                                the ESP32's own 5V pin - not enough
 *                                                current, can reset the board)
 *   IMPORTANT: tie that separate supply's GND to the ESP32's GND too (common ground),
 *              otherwise the servo signal wires won't work reliably.
 *   Orange (signal) wires - one each, NOT joined together:
 *     servo "1" (green,   closest to camera) -> ESP32 GPIO 25
 *     servo "2" (breaker)                    -> ESP32 GPIO 26
 *     servo "3" (turning)                    -> ESP32 GPIO 27
 *     servo "4" (red,     furthest from camera) -> ESP32 GPIO 14
 *   Double check this order matches how you physically installed them - if a servo
 *   fires for the wrong gate, swap the pin numbers below to match reality.
 *
 * Needs the "ESP32Servo" library (install via Arduino Library Manager - search
 * "ESP32Servo" by Kevin Harrington / madhephaestus).
 */

#include <ESP32Servo.h>

// ---------------- Belt (stepper) ----------------
const int STEP_PIN = 18; // -> DM542 PUL-
const int DIR_PIN  = 19; // -> DM542 DIR-

// TODO (placeholder): you have not measured real belt speed yet.
// Smaller number = faster belt. Each loop() toggles the pin once, and the
// driver steps on every edge, so the real step rate is 1 / (2 * stepIntervalUs).
// Tune this once you know how fast you want the belt to move, then update
// BELT_SPEED_CMS in conveyor_integration.py to match the real measured speed.
unsigned long stepIntervalUs = 800;
unsigned long lastStepTime = 0;
bool stepPinState = false;

// TODO: confirm which level makes the belt move FORWARD (toward the gates).
// If it runs backwards, just flip this to the other value.
const int DIR_FORWARD = LOW;

// ---------------- Gates (servos) ----------------
// Order must match the physical belt order: green -> breaker -> turning -> red
const int SERVO_PINS[4]      = {25, 26, 27, 14};
const char GATE_COMMANDS[4]  = {'G', 'B', 'T', 'R'};
Servo gateServos[4];

// TODO: tune these two angles to match your gate mechanism (whatever angle
// actually swings the gate open vs. lets it sit flat/closed).
const int SERVO_HOME_ANGLE = 0;
const int SERVO_OPEN_ANGLE = 90;

// How long the gate stays open before it swings back on its own.
const unsigned long GATE_OPEN_DURATION_MS = 1200;

unsigned long gateReturnAt[4] = {0, 0, 0, 0};
bool gateIsOpen[4] = {false, false, false, false};

void setup() {
  Serial.begin(115200);

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(DIR_PIN, DIR_FORWARD);

  for (int i = 0; i < 4; i++) {
    gateServos[i].setPeriodHertz(50);
    gateServos[i].attach(SERVO_PINS[i], 500, 2400);
    gateServos[i].write(SERVO_HOME_ANGLE);
  }

  Serial.println("Tomato sorter ready.");
}

void loop() {
  runBelt();
  handleSerialCommands();
  handleGateReturns();
}

// Keeps the belt moving. Non-blocking - never uses delay(), so it doesn't
// stall while a gate is opening.
void runBelt() {
  unsigned long now = micros();
  if (now - lastStepTime >= stepIntervalUs) {
    lastStepTime = now;
    stepPinState = !stepPinState;
    // Common-anode wiring: pulling PUL- low (while PUL+ is fixed at 3V3)
    // is what actually triggers the step on most DM542 setups.
    digitalWrite(STEP_PIN, stepPinState ? LOW : HIGH);
  }
}

// Reads any letters waiting on Serial and opens the matching gate.
// Unrecognized characters (stray newlines, etc.) are just ignored.
void handleSerialCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    for (int i = 0; i < 4; i++) {
      if (c == GATE_COMMANDS[i]) {
        openGate(i);
      }
    }
  }
}

void openGate(int i) {
  gateServos[i].write(SERVO_OPEN_ANGLE);
  gateIsOpen[i] = true;
  gateReturnAt[i] = millis() + GATE_OPEN_DURATION_MS;
}

// Swings any gate that has been open long enough back to home.
void handleGateReturns() {
  unsigned long now = millis();
  for (int i = 0; i < 4; i++) {
    if (gateIsOpen[i] && now >= gateReturnAt[i]) {
      gateServos[i].write(SERVO_HOME_ANGLE);
      gateIsOpen[i] = false;
    }
  }
}
