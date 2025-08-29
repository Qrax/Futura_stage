#include <Arduino.h>
#include "driver/i2s.h"

// --- Configuration ---
#define ADC_PIN_ENUM      ADC1_CHANNEL_6  // ADC1_CHANNEL_6 is GPIO 34.
// CORRECTED: Using the non-deprecated name for the attenuation level.
#define ADC_ATTENUATION   ADC_ATTEN_DB_12
#define ADC_BIT_WIDTH     ADC_WIDTH_BIT_12

// --- I2S Configuration ---
#define I2S_SAMPLE_RATE   2000000
#define I2S_PORT          I2S_NUM_0
#define I2S_DMA_BUF_SIZE  1024

// --- Burst Capture Configuration ---
#define NUM_BUFFERS_TO_SEND 1

// --- Communication with main loop ---
volatile int buffers_to_send = 0;

// --- Task for reading ADC data ---
TaskHandle_t adcReaderTaskHandle;

// Function to initialize I2S & ADC
void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = I2S_DMA_BUF_SIZE, // CORRECTED: This was the typo 'd_buf_len'
    .use_apll = true
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  adc1_config_width(ADC_BIT_WIDTH);
  adc1_config_channel_atten((adc1_channel_t)ADC_PIN_ENUM, ADC_ATTENUATION);
  i2s_set_adc_mode(ADC_UNIT_1, ADC_PIN_ENUM);
  i2s_adc_enable(I2S_PORT);
}

/**
 * @brief High-performance task to read ADC data from I2S DMA.
 */
void adcReaderTask(void *parameter) {
  uint16_t* i2s_read_buffer = (uint16_t*)malloc(I2S_DMA_BUF_SIZE * sizeof(uint16_t));
  size_t bytes_read;

  while (true) {
    i2s_read(I2S_PORT, i2s_read_buffer, I2S_DMA_BUF_SIZE * sizeof(uint16_t), &bytes_read, portMAX_DELAY);

    if (buffers_to_send > 0) {
      Serial.write((const uint8_t*)i2s_read_buffer, bytes_read);
      buffers_to_send--;
    }
  }
  free(i2s_read_buffer);
  vTaskDelete(NULL);
}

void setup() {
  Serial.begin(2000000);
  Serial.println("ESP32 High-Speed ADC Logger Initialized");
  Serial.println("Send any character to trigger 10k sample capture.");

  setupI2S();

  xTaskCreatePinnedToCore(
    adcReaderTask, "ADC Reader Task", 4096, NULL, 1, &adcReaderTaskHandle, 1);
}

void loop() {
  if (Serial.available() > 0) {
    while(Serial.available()) {
      Serial.read();
    }
    buffers_to_send = NUM_BUFFERS_TO_SEND;
  }
}