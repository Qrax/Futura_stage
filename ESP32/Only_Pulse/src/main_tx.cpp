#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_PORT I2S_NUM_0

// --- Configuration for the 40kHz tone ---
const int SINE_WAVE_FREQ_HZ = 40000;
const int SAMPLES_PER_WAVE = 50;
const int I2S_SAMPLE_RATE = SINE_WAVE_FREQ_HZ * SAMPLES_PER_WAVE;

// --- Define the new, longer pulse sequence ---
const int NUM_WAVES_IN_SEQUENCE = 10;
const int PULSE_BUFFER_SAMPLES = SAMPLES_PER_WAVE * NUM_WAVES_IN_SEQUENCE; // 50 * 100 = 5000 samples

// Buffer for the active pulse sequence. NOTE: This uses more memory (~10KB)
uint16_t pulse_buffer[PULSE_BUFFER_SAMPLES];

// A separate, smaller buffer for silence (idle state)
const int SILENCE_BUFFER_SAMPLES = 64;
uint16_t silence_buffer[SILENCE_BUFFER_SAMPLES];

// --- Mode Management ---
enum OperatingMode {
  PULSE_ON_COMMAND,
  CONTINUOUS_WAVE
};

OperatingMode currentMode = CONTINUOUS_WAVE; // Start in the default mode

// Flag to trigger the pulse in PULSE_ON_COMMAND mode.
volatile bool triggerPulse = false;

// --- Function Declarations ---
void setup_i2s_dac_mode();
void generate_pulse_waveform();
void fill_silence_buffer();
void handleSerialCommands();
void runPulseMode();
void runContinuousMode();

void setup() {
  Serial.begin(2000000);
  delay(1000);

  setup_i2s_dac_mode();

  Serial.println("Generating complex waveform... please wait.");
  generate_pulse_waveform();
  fill_silence_buffer();
  Serial.println("Waveform generation complete.");
  
  i2s_zero_dma_buffer(I2S_PORT);

  Serial.println("ESP32 is ready.");
  Serial.println("---------------------------------");
  Serial.println("Commands:");
  Serial.println("  - Press [Enter] to send a single pulse sequence (in mode A).");
  Serial.println("  - Type 'mode_a' to switch to Pulse-on-Command mode.");
  Serial.println("  - Type 'mode_b' to switch to Continuous Wave mode.");
  Serial.println("---------------------------------");
  Serial.print("Current mode: Pulse-on-Command (mode_a)\n");
}

void loop() {
  handleSerialCommands();

  switch (currentMode) {
    case PULSE_ON_COMMAND:
      runPulseMode();
      break;

    case CONTINUOUS_WAVE:
      runContinuousMode();
      break;
  }
}

void handleSerialCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "mode_a") {
      if (currentMode != PULSE_ON_COMMAND) {
        currentMode = PULSE_ON_COMMAND;
        Serial.println("Switched to Pulse-on-Command mode (mode_a). Press Enter for a pulse.");
        i2s_zero_dma_buffer(I2S_PORT);
      }
    } else if (command == "mode_b") {
      if (currentMode != CONTINUOUS_WAVE) {
        currentMode = CONTINUOUS_WAVE;
        Serial.println("Switched to Continuous Wave mode (mode_b).");
      }
    } else {
      if (currentMode == PULSE_ON_COMMAND) {
        triggerPulse = true;
      }
    }
  }
}

void runPulseMode() {
  size_t bytes_written = 0;
  if (triggerPulse) {
    Serial.println("Pulse sequence sent.");
    // Write the entire 100-wave sequence to the DAC
    i2s_write(I2S_PORT, pulse_buffer, sizeof(pulse_buffer), &bytes_written, portMAX_DELAY);
    triggerPulse = false;
  } else {
    // In the idle state, continuously send silence
    i2s_write(I2S_PORT, silence_buffer, sizeof(silence_buffer), &bytes_written, portMAX_DELAY);
  }
}

void runContinuousMode() {
  size_t bytes_written = 0;
  // Continuously loop the 100-wave sequence
  i2s_write(I2S_PORT, pulse_buffer, sizeof(pulse_buffer), &bytes_written, portMAX_DELAY);
}

void setup_i2s_dac_mode() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_DAC_BUILT_IN),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_MSB,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = true
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, NULL);
  i2s_set_dac_mode(I2S_DAC_CHANNEL_LEFT_EN);
}

// /-------------------------------------------------------------------\
// |                        MODIFIED FUNCTION                          |
// \-------------------------------------------------------------------/
/**
 * @brief Generates a 100-wave sequence where the amplitude sweeps
 *        from 40 to 100 and back down.
 */
void generate_pulse_waveform() {
  const float start_amplitude = 30.0f;
  const float end_amplitude = 95.0f;

  // Outer loop: iterate through each of the 100 waves
  for (int wave_num = 0; wave_num < NUM_WAVES_IN_SEQUENCE; wave_num++) {
    float current_amplitude;

    // Determine the amplitude for the current wave
    // It ramps up for the first half, and down for the second half
    if (wave_num < NUM_WAVES_IN_SEQUENCE / 2) {
      // Ramp up from start to end
      float progress = (float)wave_num / ((NUM_WAVES_IN_SEQUENCE / 2) - 1);
      current_amplitude = start_amplitude + (end_amplitude - start_amplitude) * progress;
    } else {
      // Ramp down from end to start
      float progress = (float)(wave_num - NUM_WAVES_IN_SEQUENCE / 2) / ((NUM_WAVES_IN_SEQUENCE / 2) - 1);
      current_amplitude = end_amplitude - (end_amplitude - start_amplitude) * progress;
    }

    // Inner loop: generate the 50 samples for the current wave
    for (int i = 0; i < SAMPLES_PER_WAVE; i++) {
      float sin_value = sinf(2.0f * PI * i / SAMPLES_PER_WAVE);
      uint8_t dac_value = (uint8_t)(127.5f + current_amplitude * sin_value);

      // Calculate the index in the master buffer
      int buffer_index = wave_num * SAMPLES_PER_WAVE + i;
      pulse_buffer[buffer_index] = dac_value << 8;
    }
  }
}

void fill_silence_buffer() {
  for (int i = 0; i < SILENCE_BUFFER_SAMPLES; i++) {
    silence_buffer[i] = 127 << 8;
  }
}