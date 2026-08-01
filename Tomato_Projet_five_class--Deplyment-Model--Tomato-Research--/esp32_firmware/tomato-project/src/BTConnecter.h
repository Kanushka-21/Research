/*
 * BTConnecter
 * -----------
 * Sends and receives whole JSON objects (instead of raw bytes) over either
 * classic Bluetooth SPP or a plain USB/UART serial link -- same
 * sendSignal()/receiveSignal() API either way, since both transports are
 * just a Stream underneath.
 *
 * Framing: each JSON object is sent as one line, terminated with '\n'.
 * receiveSignal() is non-blocking -- it just checks whether a full line has
 * arrived yet, so it's safe to call every loop() without stalling anything
 * else (belt stepping, servo timing, etc).
 *
 * Needs the "ArduinoJson" library (bblanchon/ArduinoJson, v7). Bluetooth
 * mode additionally needs a board with classic Bluetooth (original ESP32
 * WROOM/WROVER -- not S2/C3/S3).
 */

#ifndef BT_CONNECTER_H
#define BT_CONNECTER_H

#include <Arduino.h>
#include "BluetoothSerial.h"
#include <ArduinoJson.h>

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled -- pick a partition scheme / board with classic Bluetooth support.
#endif

class BTConnecter {
public:
  // deviceName is what shows up in the Windows Bluetooth pairing list
  // (only used by begin(), not beginSerial()).
  explicit BTConnecter(const char *deviceName = "TomatoSorter");

  // Starts advertising as a classic Bluetooth SPP device. Call once in setup().
  void begin();

  // Uses a physical (USB/UART) serial port as the transport instead of
  // Bluetooth -- e.g. beginSerial(Serial, 115200). Call once in setup(),
  // instead of begin(), not in addition to it.
  void beginSerial(HardwareSerial &serialPort, unsigned long baud = 115200);

  // True once a PC has actually paired/connected over SPP (Bluetooth mode).
  // Always true once beginSerial() has been called (serial mode).
  bool isConnected();

  // Serializes doc to JSON and writes it out, followed by '\n'.
  void sendSignal(const JsonDocument &doc);

  // Non-blocking: pulls in any bytes waiting on the link and checks whether
  // a full '\n'-terminated line has arrived. Returns true and fills `doc`
  // only when a complete, valid JSON message was received this call --
  // call it every loop() to avoid missing data.
  bool receiveSignal(JsonDocument &doc);

private:
  BluetoothSerial _serialBT;
  const char *_deviceName;
  String _rxBuffer;
  Stream *_port = nullptr;
  bool _usingBluetooth = false;

  static const size_t MAX_LINE_LENGTH = 512;
};

#endif // BT_CONNECTER_H
