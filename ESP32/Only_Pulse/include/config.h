#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// =================================================================
// == CONFIGURATIEBESTAND VOOR ULTRASONE PULS GENERATOR           ==
// =================================================================
// In dit bestand staan alle instellingen zoals pin definities,
// timing constanten en de waveform data.

// ==== PIN DEFINITIES ====
const int DRV_EN_PIN = 23;      // Pin om de DRV2700 driver aan/uit te zetten
const int DRV_GAIN0_PIN = 22;   // Gain controle pin 0
const int DRV_GAIN1_PIN = 21;   // Gain controle pin 1
const int DAC_PIN = 25;         // Ingebouwde DAC pin (DAC1)

// ==== I2S & PULS PARAMETERS ====
// De I2S poort die we gaan gebruiken (standaard is I2S_NUM_0)
#include "driver/i2s.h"
const i2s_port_t I2S_PORT = I2S_NUM_0;

// Sample rate voor de I2S klok. Gekozen om een perfecte 40kHz te genereren.
// 1,000,000 Hz klok / 25 samples per golf = 40,000 golven per seconde (40 kHz)
const int I2S_SAMPLE_RATE = 1000000;
const int PULSE_SAMPLE_COUNT = 25;

// ==== TIMING & WACHTZONES ====
const int QUIET_ZONE_MS = 100; // Stabilisatietijd na het aanzetten van de driver in ms

// ==== SINUS WAVEFORM DATA (LOOKUP TABLE) ====
// Vooraf berekende 8-bit (0-255) sinusgolf data voor 25 samples.
// Gecentreerd rond 128 (het nulpunt van de DAC).
const byte sinus40kHz[PULSE_SAMPLE_COUNT] = {
  128, 160, 191, 218, 239, 252, 255, 248, 231, 206, 175,
  141, 107, 75,  47,  26,  11,  4,   13,  33,  59,
  89,  119, 149, 128 // Eindigt op 128 om netjes terug te keren naar het midden
};

// =================================================================
// == TESTMODUS SCHAKELAAR                                        ==
// =================================================================
// Zet op 1 om de volledige keten MET de DRV2700 hoogspanningsdriver te activeren.
// Zet op 0 om ALLEEN het 0-3.3V DAC signaal op pin 25 te testen.
#define ACTIVATE_DRV_DRIVER 0 // <-- HIER AANPASSEN (1 = DRV AAN, 0 = ALLEEN DAC)

#endif // CONFIG_H