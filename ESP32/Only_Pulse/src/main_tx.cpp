#include <Arduino.h>
#include "driver/i2s.h"
x
#define I2S_PORT I2S_NUM_0

// Configuratie van de toon
const int SINE_WAVE_FREQ_HZ = 40000;
const int SAMPLES_PER_WAVE = 50;
const int I2S_SAMPLE_RATE = SINE_WAVE_FREQ_HZ * SAMPLES_PER_WAVE;

// Buffer voor de actieve puls (alleen de sinusgolf)
uint16_t pulse_buffer[SAMPLES_PER_WAVE];

// Een aparte, kleinere buffer voor de stilte (rusttoestand)
const int SILENCE_BUFFER_SAMPLES = 64;
uint16_t silence_buffer[SILENCE_BUFFER_SAMPLES];

// Vlaggetje om de puls te triggeren. 'volatile' is goede praktijk.
volatile bool triggerPulse = false;

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

// Genereert de sinusgolf-data en slaat deze op in de pulse_buffer
void generate_pulse_waveform() {
  for (int i = 0; i < SAMPLES_PER_WAVE; i++) {
    uint8_t dac_value = (uint8_t)(127.5 + 85 * sinf(2.0f * PI * i / SAMPLES_PER_WAVE));
    pulse_buffer[i] = dac_value << 8;
  }
}

// Vult de silence_buffer met de 1.5V rustwaarde
void fill_silence_buffer() {
  for (int i = 0; i < SILENCE_BUFFER_SAMPLES; i++) {
    silence_buffer[i] = 127 << 8;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  setup_i2s_dac_mode();

  // Maak de golfvormen één keer aan het begin
  generate_pulse_waveform();
  fill_silence_buffer();
  
  i2s_zero_dma_buffer(I2S_PORT);

  Serial.println("ESP32 is klaar.");
  Serial.println("Wacht ~0.5s tot het signaal op de scope 0V is.");
  Serial.println("Druk daarna op Enter om een puls te sturen.");
}

void loop() {
  size_t bytes_written = 0;

  // Controleer of er een commando is om de puls te sturen
  if (Serial.available() > 0) {
    while (Serial.available() > 0) {
      Serial.read();
    }
    triggerPulse = true; // Zet het vlaggetje
  }

  // State machine: stuur de puls of stuur stilte
  if (triggerPulse) {
    Serial.println("Puls wordt verstuurd...");
    i2s_write(I2S_PORT, pulse_buffer, sizeof(pulse_buffer), &bytes_written, portMAX_DELAY);
    triggerPulse = false; // Zet vlaggetje meteen uit, zodat we maar 1 puls sturen
  } else {
    // In de rusttoestand: stuur continu stilte om de 1.5V lijn vast te houden
    i2s_write(I2S_PORT, silence_buffer, sizeof(silence_buffer), &bytes_written, portMAX_DELAY);
  }
}