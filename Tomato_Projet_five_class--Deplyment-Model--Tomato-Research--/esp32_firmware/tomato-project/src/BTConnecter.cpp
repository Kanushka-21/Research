#include "BTConnecter.h"

BTConnecter::BTConnecter(const char *deviceName) : _deviceName(deviceName) {}

void BTConnecter::begin() {
  _serialBT.begin(_deviceName);
  _port = &_serialBT;
  _usingBluetooth = true;
  _rxBuffer.reserve(MAX_LINE_LENGTH);
}

void BTConnecter::beginSerial(HardwareSerial &serialPort, unsigned long baud) {
  serialPort.begin(baud);
  _port = &serialPort;
  _usingBluetooth = false;
  _rxBuffer.reserve(MAX_LINE_LENGTH);
}

bool BTConnecter::isConnected() {
  if (_usingBluetooth) {
    return _serialBT.hasClient();
  }
  return _port != nullptr;
}

void BTConnecter::sendSignal(const JsonDocument &doc) {
  if (_port == nullptr) {
    return;
  }
  String out;
  serializeJson(doc, out);
  _port->print(out);
  _port->print('\n');
}

bool BTConnecter::receiveSignal(JsonDocument &doc) {
  if (_port == nullptr) {
    return false;
  }
  while (_port->available() > 0) {
    char c = _port->read();

    if (c == '\n') {
      if (_rxBuffer.length() == 0) {
        continue; // ignore blank lines (e.g. stray \r\n)
      }
      DeserializationError err = deserializeJson(doc, _rxBuffer);
      _rxBuffer = "";
      if (err) {
        continue; // malformed line -- drop it and keep reading
      }
      return true;
    }

    if (c != '\r') {
      if (_rxBuffer.length() >= MAX_LINE_LENGTH) {
        _rxBuffer = ""; // overflow guard: drop whatever came in and resync
      }
      _rxBuffer += c;
    }
  }
  return false;
}
