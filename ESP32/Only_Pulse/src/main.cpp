#include <Arduino.h>
#include "driver/i2s.h"

// --- Configuratie (deze blijft hetzelfde) ---
const int SINE_WAVE_FREQ_HZ = 40000;
const int SAMPLES_PER_WAVE = 50;
const int I2S_SAMPLE_RATE = (SINE_WAVE_FREQ_HZ * SAMPLES_PER_WAVE);

const int WAVES_PER_BURST = 20;
const int SAMPLE_COUNT_PER_BURST = (SAMPLES_PER_WAVE * WAVES_PER_BURST);

#define I2S_PORT (I2S_NUM_0)

const int DRV_EN_PIN = 23;
const int DRV_GAIN0_PIN = 22;
const int DRV_GAIN1_PIN = 21;

uint16_t i2s_buffer[SAMPLE_COUNT_PER_BURST];

void generate_sine_wave() {
  for (int i = 0; i < SAMPLE_COUNT_PER_BURST; i++) {
    uint8_t dac_value = (uint8_t)(127.5 + 127.5 * sin(2.0 * PI * (float)i / SAMPLES_PER_WAVE));
    i2s_buffer[i] = (dac_value << 8);
  }
}

// --- HIER ZIT DE CORRECTIE ---
/**
 * @brief Configureert de I2S peripheral in DAC modus.
 */
void setup_i2s_dac_mode() {
  Serial.println("Configureren I2S in DAC modus (v2 - Corrected)...");
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_DAC_BUILT_IN),
    .sample_rate = I2S_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    // AANPASSING 1: Wees expliciet dat we alleen het rechter kanaal gebruiken.
    .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
    .communication_format = I2S_COMM_FORMAT_STAND_MSB,
    .intr_alloc_flags = 0,
    .dma_buf_count = 2,
    .dma_buf_len = 1024, // Grotere buffer kan helpen met stabiliteit
    .use_apll = false,
    .tx_desc_auto_clear = true
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, NULL); 
  
  // AANPASSING 2: Gebruik de JUISTE constante voor GPIO25 (DAC kanaal 1)
  i2s_set_dac_mode(I2S_DAC_CHANNEL_RIGHT_EN);
  
  Serial.println("...Configuratie voltooid.");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("--- I2S DAC Waveform Generator ---");

  pinMode(DRV_EN_PIN, OUTPUT);
  pinMode(DRV_GAIN0_PIN, OUTPUT);
  pinMode(DRV_GAIN1_PIN, OUTPUT);
  digitalWrite(DRV_EN_PIN, LOW);
  digitalWrite(DRV_GAIN0_PIN, LOW);
  digitalWrite(DRV_GAIN1_PIN, LOW);

  generate_sine_wave();
  setup_i2s_dac_mode();
}

void loop() {
  Serial.println("Nieuwe puls versturen via I2S/DMA...");
  
  digitalWrite(DRV_EN_PIN, HIGH);
  delay(1);

  size_t bytes_written = 0;
  // We schrijven nu de hele buffer in een keer
  i2s_write(I2S_PORT, i2s_buffer, sizeof(i2s_buffer), &bytes_written, portMAX_DELAY);
  
  digitalWrite(DRV_EN_PIN, LOW);

  Serial.printf("%d bytes geschreven naar I2S DMA.\n", bytes_written);

  delay(2000);
}