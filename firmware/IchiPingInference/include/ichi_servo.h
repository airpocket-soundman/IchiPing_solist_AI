#ifndef ICHI_SERVO_H
#define ICHI_SERVO_H

/* PCA9685 (I2C 0x40 on the shared I2CF0 bus) driving SG90 servos at 50 Hz.
   Channels: 0 = window a, 1 = window b, 2 = window c, 3 = door AB, 4 = door BC.
   OE (P42, 10 k pull-up on the interposer): High = all PWM outputs off. */

#include <stdbool.h>
#include <stdint.h>

#define ICHI_SERVO_CHANNELS  (5U)
#define ICHI_SERVO_CENTER_US (1500U)

/* Configures the PCA9685 (50 Hz, totem-pole) with every output fully off and OE High. */
bool IchiServoInitialize(void);
/* OE Low (outputs follow the PCA9685 registers) or High (all outputs off at once). */
void IchiServoEnable(bool enable);
/* Pulse width in microseconds, limited to 1000..2000 us. */
bool IchiServoSetPulse(uint8_t channel, uint16_t pulse_us);
/* Output fully off (servo unpowered by signal, stops holding). */
bool IchiServoRelease(uint8_t channel);

#endif
