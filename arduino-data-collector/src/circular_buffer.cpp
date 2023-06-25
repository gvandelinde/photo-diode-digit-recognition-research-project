#include "circular_buffer.hpp"
CircularBuffer::CircularBuffer() : bufferIndex(0) {}

int CircularBuffer::getBufferSize()
{
    return BUFFER_SIZE;
}

void CircularBuffer::addToCircularBuffer(uint16_t sample)
{
    circularBuffer[bufferIndex] = sample;
    bufferIndex = (bufferIndex + 1) % BUFFER_SIZE; // Wrap around to the beginning if the buffer is full
}

uint16_t CircularBuffer::getEarliestSample()
{
    int index = (bufferIndex + 1) % BUFFER_SIZE;
    return circularBuffer[index];
}

bool CircularBuffer::detectGestureStart(uint16_t currentSample, int threshold)
{
    uint16_t sample15StepsAgo = getEarliestSample(); 
    // Serial.print(currentSample);
    // Serial.print(" vs ");
    // Serial.print(sample15StepsAgo);
    // Serial.println();
    int difference = currentSample - sample15StepsAgo;
    if (abs(difference) > threshold)
    {
        Serial.print(currentSample);
        Serial.print(" vs ");
        Serial.print(sample15StepsAgo);
        Serial.println();
         // Serial.println("returns true");
        return true;
    }
    // Serial.println("returns false");
    return false;
}