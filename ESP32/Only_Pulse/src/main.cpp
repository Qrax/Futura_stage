#include <Arduino.h>
#include "driver/i2s.h"

// =================================================================
// ==              COMMON COMMUNICATION PINS                      ==
// =================================================================
#define M2S_TRIGGER_PIN 27 // Master-to-Slave: Request signal
#define S2M_READY_PIN   33 // Slave-to-Master: Acknowledge/Ready signal


// =================================================================
// ==              MASTER CODE BLOCK                              ==
// =================================================================
#ifdef ROLE_MASTER

// --- ADC Configuration ---
#define ADC_INPUT_PIN     ADC1_CHANNEL_6    // GPIO 34 for ADC input
#define ADC_ATTENUATION   ADC_ATTEN_DB_12
#define ADC_BIT_WIDTH     ADC_WIDTH_BIT_12

// --- I2S ADC Configuration ---
#define I2S_ADC_PORT      I2S_NUM_0
#define I2S_SAMPLE_RATE   2000000
#define I2S_DMA_BUF_LEN   1024

uint16_t adc_capture_buffer[I2S_DMA_BUF_LEN];

// Function Declarations for Master
void setup_master_pins();
void setup_i2s_adc();
void trigger_and_measure();

void setup() {
  Serial.begin(2000000);
  delay(1000);
  Serial.println("--- Master ESP32 Initialized ---");
  setup_master_pins();
  setup_i2s_adc();
  Serial.println("Ready for Python script to send trigger.");
}

void loop() {
  if (Serial.available() > 0) {
    while(Serial.available()) Serial.read(); // Clear the buffer
    trigger_and_measure();
  }
}

void setup_master_pins() {
  pinMode(M2S_TRIGGER_PIN, OUTPUT);
  pinMode(S2M_READY_PIN, INPUT_PULLDOWN);
  digitalWrite(M2S_TRIGGER_PIN, LOW);
}

void setup_i2s_adc() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 2,
    .dma_buf_len = I2S_DMA_BUF_LEN,
    .use_apll = false
  };
  i2s_driver_install(I2S_ADC_PORT, &i2s_config, 0, NULL);
  adc1_config_width(ADC_BIT_WIDTH);
  adc1_config_channel_atten(ADC_INPUT_PIN, ADC_ATTENUATION);
  i2s_set_adc_mode(ADC_UNIT_1, ADC_INPUT_PIN);
  i2s_adc_enable(I2S_ADC_PORT);
}

void trigger_and_measure() {
  Serial.println("Master: Trigger received. Requesting pulse from slave...");
  digitalWrite(M2S_TRIGGER_PIN, HIGH);

  unsigned long startTime = micros();
  while (digitalRead(S2M_READY_PIN) == LOW) {
    if (micros() - startTime > 100000) { // 100ms timeout
      Serial.println("Master: ERROR - Timed out waiting for slave.");
      digitalWrite(M2S_TRIGGER_PIN, LOW);
      return;
    }
  }

  Serial.println("Master: Slave is ready! Capturing ADC data NOW.");
  size_t bytes_read = 0;
  i2s_read(I2S_ADC_PORT, adc_capture_buffer, sizeof(adc_capture_buffer), &bytes_read, portMAX_DELAY);
  digitalWrite(M2S_TRIGGER_PIN, LOW);

  if (bytes_read > 0) {
    Serial.println("---BEGIN-DATA---");
    for (int i = 0; i < I2S_DMA_BUF_LEN; i++) {
      uint16_t sample = adc_capture_buffer[i] >> 4;
      Serial.println(sample);
    }
    Serial.println("---END-DATA---");
  } else {
    Serial.println("Master: ERROR - No data read from ADC.");
  }
  Serial.println("--- Cycle Complete. Ready for next trigger. ---");
}

#endif // ROLE_MASTER


// =================================================================
// ==              SLAVE CODE BLOCK (FINAL VERSION)               ==
// =================================================================
#ifdef ROLE_SLAVE

// --- I2S DAC Configuration ---
#define I2S_DAC_PORT I2S_NUM_0

// --- 40kHz Tone Configuration ---
const int SINE_WAVE_FREQ_HZ = 40000;
const int SAMPLES_PER_WAVE = 50;
const int I2S_SAMPLE_RATE = SINE_WAVE_FREQ_HZ * SAMPLES_PER_WAVE;

// --- Pulse Sequence Configuration (NO LEADER NEEDED ANYMORE) ---
const int NUM_WAVES_IN_SEQUENCE = 10;
const int PULSE_BUFFER_SAMPLES = SAMPLES_PER_WAVE * NUM_WAVES_IN_SEQUENCE;
uint16_t pulse_buffer[PULSE_BUFFER_SAMPLES];

// --- Buffer for continuous silence at the correct DC offset ---
const int SILENCE_BUFFER_SAMPLES = 128;
uint16_t silence_buffer[SILENCE_BUFFER_SAMPLES];

volatile bool sendPulseFlag = false;

// Function Declarations for Slave
void setup_slave_pins();
void setup_i2s_dac();
void generate_pulse_waveform(); // Back to the original name
void fill_silence_buffer();
void IRAM_ATTR handle_trigger_isr();

void setup() {
  Serial.begin(2000000);
  delay(1000);
  Serial.println("--- Slave ESP32 Initialized ---");
  setup_slave_pins();
  setup_i2s_dac();
  Serial.println("Generating waveform and silence buffer...");
  generate_pulse_waveform();
  fill_silence_buffer(); // Prepare the silence buffer
  Serial.println("Waveform ready. Waiting for trigger from master...");
  attachInterrupt(digitalPinToInterrupt(M2S_TRIGGER_PIN), handle_trigger_isr, RISING);
}

void loop() {
  size_t bytes_written = 0;
  if (sendPulseFlag) {
    // A trigger was received: send the pre-generated pulse
    i2s_write(I2S_DAC_PORT, pulse_buffer, sizeof(pulse_buffer), &bytes_written, portMAX_DELAY);
    sendPulseFlag = false;
    digitalWrite(S2M_READY_PIN, LOW); // Signal that the pulse is done
    Serial.println("Slave: Pulse sent.");
  } else {
    // In the idle state, continuously send silence at the DC offset
    // This keeps the DAC at ~1.65V, preventing any transients.
    i2s_write(I2S_DAC_PORT, silence_buffer, sizeof(silence_buffer), &bytes_written, portMAX_DELAY);
  }
}

void setup_slave_pins() {
  pinMode(M2S_TRIGGER_PIN, INPUT_PULLDOWN);
  pinMode(S2M_READY_PIN, OUTPUT);
  digitalWrite(S2M_READY_PIN, LOW);
}

void IRAM_ATTR handle_trigger_isr() {
  digitalWrite(S2M_READY_PIN, HIGH);
  sendPulseFlag = true;
}

void fill_silence_buffer() {
  uint16_t silence_value = 127 << 8; // The ~1.65V DC offset
  for (int i = 0; i < SILENCE_BUFFER_SAMPLES; i++) {
    silence_buffer[i] = silence_value;
  }
}

void setup_i2s_dac() {
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
  i2s_driver_install(I2S_DAC_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_DAC_PORT, NULL);
  i2s_set_dac_mode(I2S_DAC_CHANNEL_LEFT_EN);
}

// This is the original pulse generation, WITHOUT the leader.
void generate_pulse_waveform() {
  const float start_amplitude = 30.0f;
  const float end_amplitude = 95.0f;
  for (int wave_num = 0; wave_num < NUM_WAVES_IN_SEQUENCE; wave_num++) {
    float current_amplitude;
    if (wave_num < NUM_WAVES_IN_SEQUENCE / 2) {
      float progress = (float)wave_num / ((NUM_WAVES_IN_SEQUENCE / 2) - 1);
      current_amplitude = start_amplitude + (end_amplitude - start_amplitude) * progress;
    } else {
      float progress = (float)(wave_num - NUM_WAVES_IN_SEQUENCE / 2) / ((NUM_WAVES_IN_SEQUENCE / 2) - 1);
      current_amplitude = end_amplitude - (end_amplitude - start_amplitude) * progress;
    }
    for (int i = 0; i < SAMPLES_PER_WAVE; i++) {
      float sin_value = sinf(2.0f * PI * i / SAMPLES_PER_WAVE);
      uint8_t dac_value = (uint8_t)(127.5f + current_amplitude * sin_value);
      int buffer_index = (wave_num * SAMPLES_PER_WAVE) + i;
      pulse_buffer[buffer_index] = dac_value << 8;
    }
  }
}

#endif // ROLE_SLAVE