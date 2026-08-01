/*
 * DataModel
 * ---------
 * One command + one value, e.g. {"command":"servo1","value":90.0}.
 *
 * `command` is restricted to the fixed set of actuators/sensors below --
 * setCommand()/fromJson() reject anything else so a typo or a stray
 * Bluetooth byte never turns into a silent no-op on the wrong pin.
 */

#ifndef DATA_MODEL_H
#define DATA_MODEL_H

#include <Arduino.h>
#include <ArduinoJson.h>

class DataModel {
public:
  DataModel();
  DataModel(const String &command, float value);

  // Returns false (and leaves the command unchanged) if `command` isn't
  // one of the allowed values.
  bool setCommand(const String &command);
  String getCommand() const;

  void setValue(float value);
  float getValue() const;

  // True once a valid command has been set.
  bool isValid() const;

  void toJson(JsonDocument &doc) const;

  // Parses doc["command"] / doc["value"]. Returns false if the command is
  // missing or not one of the allowed values -- `out` is left untouched
  // in that case.
  static bool fromJson(const JsonDocument &doc, DataModel &out);

  static bool isValidCommand(const String &command);

private:
  String _command;
  float _value;
  bool _valid;
};

#endif // DATA_MODEL_H
