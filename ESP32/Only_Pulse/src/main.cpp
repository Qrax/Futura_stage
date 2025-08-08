#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_PORT I2S_NUM_0

// Tone configuration
const int SINE_WAVE_FREQ_HZ = 40000;
const int SAMPLES_PER_WAVE = 50;
const int I2S_SAMPLE_RATE = SINE_WAVE_FREQ_HZ * SAMPLES_PER_WAVE;

// Buffer for one sine wave cycle
uint16_t single_wave_buffer[SAMPLES_PER_WAVE];

void setup_i2s_dac_mode() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_DAC_BUILT_IN),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // match left DAC
    .communication_format = I2S_COMM_FORMAT_STAND_MSB,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = true
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, NULL);
  i2s_set_dac_mode(I2S_DAC_CHANNEL_LEFT_EN); // GPIO25
}

void generate_single_wave() {
  for (int i = 0; i < SAMPLES_PER_WAVE; i++) {
    uint8_t dac_value = (uint8_t)(127.5 + 85 *
                                  sinf(2.0f * PI * i / SAMPLES_PER_WAVE));
    single_wave_buffer[i] = dac_value << 8; // 8-bit DAC value in MSB
  }
}

void setup() {
  setup_i2s_dac_mode();
  generate_single_wave();
}

void loop() {
  size_t bytes_written;
  // Continuously output the sine wave
  i2s_write(I2S_PORT, single_wave_buffer, sizeof(single_wave_buffer),
            &bytes_written, portMAX_DELAY);
}
