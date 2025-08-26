#include <Arduino.h>
#include "driver/i2s.h"
#include "driver/adc.h" // Required for ADC2 functions
#include "esp_adc_cal.h"

// =================================================================
// ==                  ADC/I2S CONFIGURATION                    ==
// =================================================================
// --- SWITCHING TO ADC2 ---
#define ADC_UNIT          ADC_UNIT_2
#define ADC_CHANNEL       ADC2_CHANNEL_8 // This is GPIO 25
#define ADC_PIN           25
// -------------------------

#define ADC_ATTENUATION   ADC_ATTEN_DB_12
#define ADC_BIT_WIDTH     ADC_WIDTH_BIT_12
#define I2S_ADC_PORT      I2S_NUM_0
#define I2S_SAMPLE_RATE   500000
#define I2S_DMA_BUF_LEN   1024

// =================================================================
// ==                     TEST PARAMETERS                         ==
// =================================================================
const unsigned long TEST_INTERVAL_MS = 2000;
unsigned long previousMillis = 0;

// =================================================================
// ==                  GLOBAL VARIABLES                           ==
// =================================================================
uint16_t i2s_read_buffer[I2S_DMA_BUF_LEN];
esp_adc_cal_characteristics_t adc_chars;

// Forward declarations
void setup_i2s_adc_final();
void setup_calibration();
void perform_adc_test();

// =================================================================
// ==                        SETUP                                ==
// =================================================================
void setup() {
  Serial.begin(2000000);
  delay(1000);
  Serial.println("\n--- ADC Test: Final Version using ADC2 on GPIO 25 ---");

  // It's still good practice to disable WiFi
  WiFi.mode(WIFI_OFF);
  btStop();

  setup_i2s_adc_final();
  setup_calibration();

  Serial.printf("\nSetup complete. Connect your voltage source to GPIO %d.\n", ADC_PIN);
  Serial.println("Starting measurements in 3 seconds...");
  delay(3000);
}

// =================================================================
// ==                         LOOP                                ==
// =================================================================
void loop() {
  if (millis() - previousMillis >= TEST_INTERVAL_MS) {
    previousMillis = millis();
    perform_adc_test();
  }
}

// =================================================================
// ==                  HELPER FUNCTIONS                           ==
// =================================================================

void setup_i2s_adc_final() {
  // ADC2 configuration is slightly different - we use adc2_config_channel_atten
  adc2_config_channel_atten(ADC_CHANNEL, ADC_ATTENUATION);

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
  i2s_set_adc_mode(ADC_UNIT, ADC_CHANNEL);
  
  Serial.println("I2S ADC driver configured on ADC2.");
}

void setup_calibration() {
  esp_adc_cal_characterize(ADC_UNIT, ADC_ATTENUATION, ADC_BIT_WIDTH, 1100, &adc_chars);
}

void perform_adc_test() {
  size_t bytes_read = 0;
  
  i2s_adc_enable(I2S_ADC_PORT);
  i2s_read(I2S_ADC_PORT, i2s_read_buffer, sizeof(i2s_read_buffer), &bytes_read, portMAX_DELAY);
  i2s_adc_disable(I2S_ADC_PORT);

  if (bytes_read == 0) {
    Serial.println("Error: Failed to read from I2S ADC.");
    return;
  }

  long raw_sum = 0;
  uint16_t raw_min = 4095;
  uint16_t raw_max = 0;

  for (int i = 0; i < I2S_DMA_BUF_LEN; i++) {
    uint16_t raw_sample = i2s_read_buffer[i] & 0x0FFF; // Mask for 12 bits
    raw_sum += raw_sample;
    if (raw_sample < raw_min) raw_min = raw_sample;
    if (raw_sample > raw_max) raw_max = raw_sample;
  }

  float raw_avg = (float)raw_sum / I2S_DMA_BUF_LEN;
  uint32_t voltage_avg = esp_adc_cal_raw_to_voltage(raw_avg, &adc_chars);

  Serial.println("\n--- New Measurement ---");
  Serial.println("--- RAW ADC VALUES (0-4095) ---");
  Serial.printf("Average: %.2f\n", raw_avg);
  Serial.printf("Min:     %u\n", raw_min);
  Serial.printf("Max:     %u\n\n", raw_max);
  Serial.println("--- CALIBRATED VOLTAGE (mV) ---");
  Serial.printf("Average: %u mV\n", voltage_avg);
  Serial.println("---------------------------------");
}