#include <iostream>
#include <Arduino.h>

#ifndef CIRCULAR_BUFFER
#define CIRCULAR_BUFFER

class CircularBuffer
{
public:
    CircularBuffer();

    int getBufferSize();
    void addToCircularBuffer(uint16_t sample);
    uint16_t getEarliestSample();
    bool detectGestureStart(u_int16_t currentSample, int treshold);

private:
    static const int BUFFER_SIZE = 100;
    uint16_t circularBuffer[BUFFER_SIZE];
    int bufferIndex;
};

#endif // CIRCULAR_BUFFER