#include <Arduino.h>
#include "driver/i2s.h"

// =================================================================
// ==              COMMON COMMUNICATION PINS                      ==
// =================================================================
#define M2S_TRIGGER_PIN 27 // Master-to-Slave: Request signal
#define S2M_READY_PIN   33 // Slave-to-Master: Acknowledge/Ready signal


// =================================================================
// ==           MASTER CODE BLOCK (WITH DEBUG SIGNAL)             ==
// =================================================================
#ifdef ROLE_MASTER

#define ADC_INPUT_PIN     ADC1_CHANNEL_0
#define ADC_ATTENUATION   ADC_ATTEN_DB_12
#define ADC_BIT_WIDTH     ADC_WIDTH_BIT_12
#define I2S_ADC_PORT      I2S_NUM_0
#define I2S_SAMPLE_RATE   400000
#define I2S_DMA_BUF_LEN   1024
#define TOTAL_SAMPLES     1024

// TUNE THIS VALUE: The delay to account for the speed of sound.
#define LISTENING_DELAY_MICROSECONDS 60

// NEW: A pin to visualize the ADC capture window on an oscilloscope
#define MASTER_DEBUG_PIN 21

uint16_t adc_capture_buffer[I2S_DMA_BUF_LEN];

void setup_master_pins();
void setup_i2s_adc();
void trigger_and_measure();

void setup() {
  Serial.begin(921600);
  delay(1000);
  Serial.println("--- Master ESP32 Initialized ---");
  setup_master_pins();
  setup_i2s_adc();
  Serial.println("Ready for Python script to send trigger.");
}

void loop() {
  if (Serial.available() > 0) {
    while(Serial.available()) Serial.read();
    trigger_and_measure();
  }
}

void setup_master_pins() {
  pinMode(M2S_TRIGGER_PIN, OUTPUT);
  pinMode(S2M_READY_PIN, INPUT_PULLDOWN);
  digitalWrite(M2S_TRIGGER_PIN, LOW);
  
  // NEW: Setup the debug pin
  pinMode(MASTER_DEBUG_PIN, OUTPUT);
  digitalWrite(MASTER_DEBUG_PIN, LOW);
}

void setup_i2s_adc() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = I2S_DMA_BUF_LEN,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_driver_install(I2S_ADC_PORT, &i2s_config, 0, NULL);
  adc1_config_width(ADC_BIT_WIDTH);
  adc1_config_channel_atten(ADC_INPUT_PIN, ADC_ATTEN_DB_12);
  i2s_set_adc_mode(ADC_UNIT_1, ADC_INPUT_PIN);
  //i2s_adc_enable(I2S_ADC_PORT);
}

void trigger_and_measure() {
  Serial.println("Master: Trigger received. Requesting pulse...");
  digitalWrite(M2S_TRIGGER_PIN, HIGH);

  unsigned long startTime = micros();
  while (digitalRead(S2M_READY_PIN) == LOW) {
    if (micros() - startTime > 100000) {
      Serial.println("Master: ERROR - Timed out waiting for slave.");
      digitalWrite(M2S_TRIGGER_PIN, LOW);
      return;
    }
  }
  
  delayMicroseconds(LISTENING_DELAY_MICROSECONDS);

  Serial.println("Master: Propagation delay finished. Capturing ADC...");
  
  // --- DEBUG SIGNAL GOES HIGH ---
  digitalWrite(MASTER_DEBUG_PIN, HIGH);
  
  size_t bytes_read = 0;


  i2s_zero_dma_buffer(I2S_ADC_PORT);
  i2s_adc_enable(I2S_ADC_PORT);

  i2s_read(I2S_ADC_PORT, adc_capture_buffer, sizeof(adc_capture_buffer), &bytes_read, portMAX_DELAY);
  
  while(bytes_read < TOTAL_SAMPLES * sizeof(uint16_t)) {
    size_t bytes_read_now = 0;
    i2s_read(I2S_ADC_PORT, ((uint8_t*)adc_capture_buffer) + bytes_read, sizeof(adc_capture_buffer) - bytes_read, &bytes_read_now, portMAX_DELAY);
    if (bytes_read_now > 0) {
      bytes_read += bytes_read_now;
    }
  }

  i2s_adc_disable(I2S_ADC_PORT);
  // --- DEBUG SIGNAL GOES LOW ---
  digitalWrite(MASTER_DEBUG_PIN, LOW);
  
  digitalWrite(M2S_TRIGGER_PIN, LOW);

  if (bytes_read > 0) {
    Serial.println("---BEGIN-DATA---");
    for (int i = 0; i < I2S_DMA_BUF_LEN; i++) {
      uint16_t sample = adc_capture_buffer[i];// >> 4;
      Serial.println(sample);
    }
    Serial.println("---END-DATA---");
  } else {
    Serial.println("Master: ERROR - No data read from ADC.");
  }
  Serial.println("--- Cycle Complete ---");
}

#endif // ROLE_MASTER


// =================================================================
// ==           SLAVE CODE BLOCK (STABLE & CORRECT)               ==
// =================================================================
#ifdef ROLE_SLAVE

#define I2S_DAC_PORT I2S_NUM_0

const int SINE_WAVE_FREQ_HZ = 40000;
const int SAMPLES_PER_WAVE = 50;
const int I2S_SAMPLE_RATE = SINE_WAVE_FREQ_HZ * SAMPLES_PER_WAVE;

const int NUM_WAVES_IN_SEQUENCE = 10;
const int PULSE_BUFFER_SAMPLES = SAMPLES_PER_WAVE * NUM_WAVES_IN_SEQUENCE;
const int PULSE_DURATION_MICROSECONDS = PULSE_BUFFER_SAMPLES * 1000000 / I2S_SAMPLE_RATE;

uint16_t pulse_buffer[PULSE_BUFFER_SAMPLES];
const int SILENCE_BUFFER_SAMPLES = 128;
uint16_t silence_buffer[SILENCE_BUFFER_SAMPLES];
volatile bool sendPulseFlag = false;

void setup_slave_pins();
void setup_i2s_dac();
void generate_pulse_waveform();
void fill_silence_buffer();
void IRAM_ATTR handle_trigger_isr();

void setup() {
  Serial.begin(2000000);
  delay(1000);
  Serial.println("--- Slave ESP32 Initialized ---");
  setup_slave_pins();
  setup_i2s_dac();
  generate_pulse_waveform();
  fill_silence_buffer();
  Serial.println("Waveform ready. Waiting for trigger...");
  attachInterrupt(digitalPinToInterrupt(M2S_TRIGGER_PIN), handle_trigger_isr, RISING);
}

void loop() {
  size_t bytes_written = 0;
  if (sendPulseFlag) {
    i2s_zero_dma_buffer(I2S_DAC_PORT);
    i2s_write(I2S_DAC_PORT, pulse_buffer, sizeof(pulse_buffer), &bytes_written, portMAX_DELAY);
    delayMicroseconds(PULSE_DURATION_MICROSECONDS);
    sendPulseFlag = false;
    digitalWrite(S2M_READY_PIN, LOW);
    Serial.println("Slave: Pulse sent.");
  } else {
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
  uint16_t silence_value = 127 << 8;
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