#include <Arduino.h>
#include "diode_calibration.hpp"
#include "circular_buffer.hpp"
#include "leds.hpp"
#include "map"

// Uncomment this to get binary responses for photodiode values.
#define BINARY_RESPONE 0x00

LightIntensityRegulator *regulator;

// Sample rate in hertz
// Can be changed over the serial interface.
// Maximum is around 1200 Hz.
uint16_t SAMPLE_RATE = 1000;
int TOTAL_SAMPLES = 2000;
uint32_t SAMPLE_RATE_DELAY_MICROS = 1000000 / SAMPLE_RATE;
int THRESHOLD;

struct DiodesReading {
  u_int16_t r0;
  u_int16_t r1;
  u_int16_t r2;
};

void measurementCommand();

// Helper funtion to read and return a value from the serial.
// Wrap it into its own type (template)
template <typename T>
void getValueFromSerial(T *buffer)
{

  // Wait until the right amount of bytes are available.
  // while (Serial.available() < (int) sizeof(T));

  // Read the amount of bytes that we need.
  // And store them in the buffer.
  Serial.readBytes((char *)buffer, sizeof(T));
}

// Helper function to read the photodiodes at a given sample rate.
// Returns the results over the serial interface.
DiodesReading readPhotodiodes()
{
  const unsigned long start = micros();

  DiodesReading reading;
  reading.r0 = (uint16_t)analogRead(A0);
  reading.r1 = (uint16_t)analogRead(A1);
  reading.r2 = (uint16_t)analogRead(A2);

  // Delay only if we have time left.
  const unsigned long diff = micros() - start + 4; // Add offset to compensate if statement
  if (diff < SAMPLE_RATE_DELAY_MICROS) {
    delayMicroseconds(SAMPLE_RATE_DELAY_MICROS - diff);
  }
  return reading;
}

void binary_response(DiodesReading data) {
    uint16_t r0 = data.r0;
    uint16_t r1 = data.r1;
    uint16_t r2 = data.r2;

    Serial.write((char*) &r0, sizeof(uint16_t));
    Serial.write((char*) &r1, sizeof(uint16_t));
    Serial.write((char*) &r2, sizeof(uint16_t));
    return;
}

void fillBuffers(CircularBuffer* buffers, int size) {
  // Initially fill buffer
  for (int i = 0; i < buffers[0].getBufferSize(); i++)
  {
    DiodesReading reading = readPhotodiodes();
    buffers[0].addToCircularBuffer(reading.r0);
    buffers[1].addToCircularBuffer(reading.r1);
    buffers[2].addToCircularBuffer(reading.r2);
  }

  return;
}

const char MEASUREMENT_DETECTION_LOOP = 0xAA;
void measurementLoop()
{
  setLedGreen();
  // Receive Threshold
  delay(100);

  getValueFromSerial(&THRESHOLD);

  // Board is ready for collection
  setLedBlue();

  // Create circular buffers of size 15, one for each photo diode
  CircularBuffer circular_buffers[] = {
      CircularBuffer(),
      CircularBuffer(),
      CircularBuffer()
  };

  delay(500);

  // Initially fill buffer
  fillBuffers(circular_buffers, 3); 
  // Now do loop for filling and call measurement whenever we detect a starting gesture.
  while (true)
  {
    DiodesReading reading = readPhotodiodes();
    // Detect gesture start.
    if (circular_buffers[0].detectGestureStart(reading.r0, THRESHOLD) || circular_buffers[1].detectGestureStart(reading.r1, THRESHOLD) || circular_buffers[2].detectGestureStart(reading.r2, THRESHOLD)) {
      // Signal start of measurement.
      u_int16_t cmd = 0xAA;
      Serial.write((char*) &cmd, sizeof(uint16_t));

      // Start measurement
      measurementCommand();
      fillBuffers(circular_buffers, 3); 
      delay(200);
    }
    else {
      // Update circular buffers
      circular_buffers[0].addToCircularBuffer(reading.r0);
      circular_buffers[1].addToCircularBuffer(reading.r1);
      circular_buffers[2].addToCircularBuffer(reading.r2);
    }
 }
}

// Command start of a measurement.
// Expects 4 bytes (uint32_t) that represent the amount of samples to be returned
// Uses the sample rate that is currently set.
const char MEASUREMENT_START = 0xAB;
void measurementCommand()
{
  setLedRed();
  // Collect the amount of samples to be taken.
  uint32_t samples = TOTAL_SAMPLES;
  // Read the photodiodes for the amount of samples.
  for (uint32_t i = 0; i < samples; i++)
  {
    binary_response(readPhotodiodes());
  }
  setLedBlue();
}

// Command recalibration of resistor values.
const char RECALIBRATE = 0xAC;
void recalibrateCommand()
{
  setLedOrange();
  // regulator->reconfigure();
  // Create a new light regulator.
  regulator = new LightIntensityRegulator();

  // Return the resistance that has been set.
  Serial.println(regulator->get_resistance());
}

// Set the sample rate. (in Hertz)
// Expects 2 bytes (uint16_t) that represent the sample rate.
const char SET_SAMPLE_RATE = 0xAD;
void setSampleRateCommand()
{
  setLedBlue();

  // Wait for sample rate to be available.
  uint16_t sample_rate = 0;
  getValueFromSerial(&sample_rate);

  // Set the sample rate.
  SAMPLE_RATE = sample_rate;
  SAMPLE_RATE_DELAY_MICROS = 1000000 / SAMPLE_RATE;

  Serial.print("Sample rate set to: ");
  Serial.print(sample_rate);
  Serial.println(" Hz");
}

// Make a map that contains the different commands that we can receive from the serial.
// and the functions that we should call when we receive them.
typedef void (*command_function)();
typedef std::map<char, command_function> command_map;
command_map commands =
    {
        {MEASUREMENT_DETECTION_LOOP, measurementLoop},
        {MEASUREMENT_START, measurementCommand},
        {RECALIBRATE, recalibrateCommand},
        {SET_SAMPLE_RATE, setSampleRateCommand}};

// Function that processes a command that we received from the serial.
// If the command is not in the map, we print an error message.
void processCommand(char command)
{
  // setLedPurple();
  // delay(1000);

  // Get the function
  command_function function = commands[command];

  // if the function is not null, call it.
  if (function != NULL)
  {
    function();
    return;
  }

  // If reaching here we received an unknown command
  Serial.println("Received unknown command from serial.");
}

// Function that reads the serial input and interprets it as a command.
void waitForCommand()
{

  // Set led to white to indicate that we are waiting for a command.
  setLedWhite();

  while (Serial.available() >= 1)
  {
    // Read the command from the serial.
    char command = Serial.read();
    processCommand(command);
  }
}

void setup()
{
  regulator = new LightIntensityRegulator();
  // delay(1000);
  // Serial.print("Resistance is: ");
  // Serial.println(regulator->get_resistance());
  // Initialize LED
  setupLeds();
}

// Loop that runs continuously.
// It essentially waits for a command from the serial and processes it.
void loop()
{
  waitForCommand();
  //TODO: alert, this needs to be uncommented for final deployment
  // waitForCommand();
}
