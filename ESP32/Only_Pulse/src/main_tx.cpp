#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_PORT I2S_NUM_0

// --- Configuration for the 40kHz tone ---
const int SINE_WAVE_FREQ_HZ = 40000;
const int SAMPLES_PER_WAVE = 50;
const int I2S_SAMPLE_RATE = SINE_WAVE_FREQ_HZ * SAMPLES_PER_WAVE;

// Buffer for the active pulse (the sine wave)
uint16_t pulse_buffer[SAMPLES_PER_WAVE];

// A separate, smaller buffer for silence (idle state)
const int SILENCE_BUFFER_SAMPLES = 64;
uint16_t silence_buffer[SILENCE_BUFFER_SAMPLES];

// --- Mode Management ---
enum OperatingMode {
  PULSE_ON_COMMAND, // Default mode: send one pulse when triggered
  CONTINUOUS_WAVE   // New mode: continuously send the 40kHz signal
};

OperatingMode currentMode = CONTINUOUS_WAVE; // Start in the default mode

// Flag to trigger the pulse in PULSE_ON_COMMAND mode. 'volatile' is good practice.
volatile bool triggerPulse = false;

// --- Function Declarations ---
void setup_i2s_dac_mode();
void generate_pulse_waveform();
void fill_silence_buffer();
void handleSerialCommands();
void runPulseMode();
void runContinuousMode();

void setup() {
  Serial.begin(115200);
  delay(1000);

  setup_i2s_dac_mode();

  // Generate the waveforms once at the beginning
  generate_pulse_waveform();
  fill_silence_buffer();
  
  i2s_zero_dma_buffer(I2S_PORT);

  Serial.println("ESP32 is ready.");
  Serial.println("---------------------------------");
  Serial.println("Commands:");
  Serial.println("  - Press [Enter] to send a single pulse (in mode A).");
  Serial.println("  - Type 'mode_a' to switch to Pulse-on-Command mode.");
  Serial.println("  - Type 'mode_b' to switch to Continuous 40kHz Wave mode.");
  Serial.println("---------------------------------");
  Serial.print("Current mode: Pulse-on-Command (mode_a)\n");
}

void loop() {
  // First, check for any incoming commands to change modes or trigger a pulse
  handleSerialCommands();

  // Execute behavior based on the current mode
  switch (currentMode) {
    case PULSE_ON_COMMAND:
      runPulseMode();
      break;

    case CONTINUOUS_WAVE:
      runContinuousMode();
      break;
  }
}

/**
 * @brief Checks for and processes commands from the Serial Monitor.
 */
void handleSerialCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove whitespace and newline characters

    if (command == "mode_a") {
      if (currentMode != PULSE_ON_COMMAND) {
        currentMode = PULSE_ON_COMMAND;
        Serial.println("Switched to Pulse-on-Command mode (mode_a). Press Enter for a pulse.");
        // Clear the I2S buffer to ensure we start with silence
        i2s_zero_dma_buffer(I2S_PORT);
      }
    } else if (command == "mode_b") {
      if (currentMode != CONTINUOUS_WAVE) {
        currentMode = CONTINUOUS_WAVE;
        Serial.println("Switched to Continuous Wave mode (mode_b). Generating 40kHz signal.");
      }
    } else {
      // If any other text (or just Enter) is received, treat it as a pulse trigger
      // but only if we are in the correct mode.
      if (currentMode == PULSE_ON_COMMAND) {
        triggerPulse = true;
      }
    }
  }
}

/**
 * @brief Logic for the PULSE_ON_COMMAND mode. Sends silence unless a pulse is triggered.
 */
void runPulseMode() {
  size_t bytes_written = 0;
  if (triggerPulse) {
    Serial.println("Pulse sent.");
    i2s_write(I2S_PORT, pulse_buffer, sizeof(pulse_buffer), &bytes_written, portMAX_DELAY);
    triggerPulse = false; // Reset the flag immediately to send only one pulse
  } else {
    // In the idle state, continuously send silence to maintain the 1.65V line
    i2s_write(I2S_PORT, silence_buffer, sizeof(silence_buffer), &bytes_written, portMAX_DELAY);
  }
}

/**
 * @brief Logic for the CONTINUOUS_WAVE mode. Constantly sends the sine wave.
 */
void runContinuousMode() {
  size_t bytes_written = 0;
  // Continuously write the sine wave buffer to the DAC
  i2s_write(I2S_PORT, pulse_buffer, sizeof(pulse_buffer), &bytes_written, portMAX_DELAY);
}

// --- Helper Functions (Unchanged) ---

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

// Generates the sine wave data and stores it in the pulse_buffer
void generate_pulse_waveform() {
  for (int i = 0; i < SAMPLES_PER_WAVE; i++) {
    // Generate a 8-bit sine wave (0-255) centered at 127
    // The DAC voltage is Vout = VDD * (dac_value / 255)
    // Centering at 127 gives a DC offset of ~1.65V (VDD/2)
    uint8_t dac_value = (uint8_t)(127.5 + 85 * sinf(2.0f * PI * i / SAMPLES_PER_WAVE));
    // The I2S DAC on ESP32 uses the top 8 bits of a 16-bit sample
    pulse_buffer[i] = dac_value << 8;
  }
}

// Fills the silence_buffer with the 1.65V idle value
void fill_silence_buffer() {
  for (int i = 0; i < SILENCE_BUFFER_SAMPLES; i++) {
    silence_buffer[i] = 127 << 8;
  }
}