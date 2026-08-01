#include "DataModel.h"

static const char *ALLOWED_COMMANDS[] = {
    "servo1", "servo2", "servo3", "servo4",
    "motor",
    "IR1", "IR2", "IR3", "IR4"};
static const size_t ALLOWED_COMMANDS_COUNT = sizeof(ALLOWED_COMMANDS) / sizeof(ALLOWED_COMMANDS[0]);

DataModel::DataModel() : _command(""), _value(0.0f), _valid(false) {}

DataModel::DataModel(const String &command, float value) : _command(""), _value(0.0f), _valid(false) {
  setCommand(command);
  setValue(value);
}

bool DataModel::isValidCommand(const String &command) {
  for (size_t i = 0; i < ALLOWED_COMMANDS_COUNT; i++) {
    if (command == ALLOWED_COMMANDS[i]) {
      return true;
    }
  }
  return false;
}

bool DataModel::setCommand(const String &command) {
  if (!isValidCommand(command)) {
    return false;
  }
  _command = command;
  _valid = true;
  return true;
}

String DataModel::getCommand() const {
  return _command;
}

void DataModel::setValue(float value) {
  _value = value;
}

float DataModel::getValue() const {
  return _value;
}

bool DataModel::isValid() const {
  return _valid;
}

void DataModel::toJson(JsonDocument &doc) const {
  doc["command"] = _command;
  doc["value"] = _value;
}

bool DataModel::fromJson(const JsonDocument &doc, DataModel &out) {
  if (!doc["command"].is<const char *>()) {
    return false;
  }
  String command = doc["command"].as<String>();
  if (!isValidCommand(command)) {
    return false;
  }
  out._command = command;
  out._value = doc["value"] | 0.0f;
  out._valid = true;
  return true;
}
