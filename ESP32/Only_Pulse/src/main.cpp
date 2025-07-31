#include <Arduino.h>
#include "driver/i2s.h"
#include "driver/adc.h"

// --- Configuratie ---
#define ADC_PIN           36
#define ADC_CHANNEL       ADC1_CHANNEL_0 // GPIO 36 is ADC1_CH0
#define I2S_PORT          I2S_NUM_0

// De sample rate die we AANVRAGEN. De hardware zal proberen dit te benaderen.
// Voor een 40kHz signaal is >80kS/s nodig. We proberen 100kS/s.
#define TARGET_SAMPLE_RATE (400000) // 2MHz is de maximale sample rate die we kunnen vragen

// We lezen in chunks/blokken. 1024 is een efficiënte grootte.
#define SAMPLES_PER_CHUNK  1024
// We lezen meerdere chunks achter elkaar voor een nauwkeurige meting.
#define NUM_CHUNKS         100 // Totaal = 1024 * 100 = 102.400 samples

// Buffer om ÉÉN chunk in op te slaan
uint16_t i2s_read_buffer[SAMPLES_PER_CHUNK];


void setup_adc_i2s() {
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
      .sample_rate = TARGET_SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = SAMPLES_PER_CHUNK, // DMA buffer even groot als onze chunk
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_11);
  i2s_set_adc_mode(ADC_UNIT_1, ADC_CHANNEL);

  Serial.println("ADC en I2S geconfigureerd.");
  Serial.printf("Doel sample rate: %d S/s\n", TARGET_SAMPLE_RATE);
  Serial.printf("Meting zal %d chunks van %d samples uitvoeren (totaal %d samples).\n\n", NUM_CHUNKS, SAMPLES_PER_CHUNK, NUM_CHUNKS * SAMPLES_PER_CHUNK);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  setup_adc_i2s();
}

void loop() {
  Serial.println("--- Start nieuwe, nauwkeurige meting ---");
  
  size_t bytes_read;
  long total_samples_measured = NUM_CHUNKS * SAMPLES_PER_CHUNK;

  // Start de timer
  long startTime = micros();

  // Lees het gespecificeerde aantal chunks achter elkaar
  for (int i = 0; i < NUM_CHUNKS; i++) {
    esp_err_t result = i2s_read(I2S_PORT, &i2s_read_buffer, sizeof(i2s_read_buffer), &bytes_read, portMAX_DELAY);
    if (result != ESP_OK) {
        Serial.printf("Fout tijdens lezen van chunk %d: %s\n", i, esp_err_to_name(result));
        return; // Stop de test bij een fout
    }
  }

  // Stop de timer
  long endTime = micros();

  long duration = endTime - startTime;
  float actual_sample_rate = (float)total_samples_measured / duration * 1000000.0f;

  Serial.printf("Tijd om %ld samples te meten: %ld microseconden (%.2f seconden).\n", total_samples_measured, duration, duration / 1000000.0f);
  Serial.printf("==> Werkelijke, gemeten sample rate: %.2f Samples/seconde\n", actual_sample_rate);
  
  // Wacht 3 seconden voor de volgende meting
  Serial.println("\n--- Wachten voor 3 seconden... ---\n");
  delay(3000);
}