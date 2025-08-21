#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_adc_cal.h"

// Je platformio.ini definieert ofwel ROLE_MASTER ofwel ROLE_SLAVE.
// De code hieronder selecteert automatisch het juiste blok.

// ===== Gemeenschappelijke instellingen =====
#define TRIGGER_PIN 27


// ====================================================================
// =================== CODE VOOR DE MASTER ============================
// ====================================================================
#ifdef ROLE_MASTER

#define ADC_CHANNEL ADC1_CHANNEL_6 // GPIO34
// FIX: Sample rate moet matchen met de output van de Slave (40kHz * 50 samples)
#define SAMPLE_RATE 2000000
#define BUFFER_SAMPLES 4096

uint16_t adc_buffer[BUFFER_SAMPLES];
esp_adc_cal_characteristics_t *adc_chars;

void setup_adc() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    // FIX: ADC is een mono-apparaat, dus gebruik een mono-formaat
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = true,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_adc_mode(ADC_UNIT_1, ADC_CHANNEL);
  i2s_adc_enable(I2S_NUM_0);
}

void setup() {
  Serial.begin(2000000);
  delay(200);

  pinMode(TRIGGER_PIN, OUTPUT);
  digitalWrite(TRIGGER_PIN, LOW);

  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_11);
  adc_chars = (esp_adc_cal_characteristics_t *)calloc(1, sizeof(esp_adc_cal_characteristics_t));
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, adc_chars);

  // FIX: Initialiseer de I2S driver EENMALIG in setup() voor stabiliteit en efficiëntie
  setup_adc();

  Serial.println("MASTER ready");
}

void loop() {
  delay(5000); // elke seconde trigger

  // Leeg de DMA buffer om klaar te zijn voor de nieuwe puls
  i2s_zero_dma_buffer(I2S_NUM_0);

  // Trigger sturen
  digitalWrite(TRIGGER_PIN, HIGH);
  delayMicroseconds(10); // kort hoog
  digitalWrite(TRIGGER_PIN, LOW);

  // Lees de data die direct na de trigger is opgenomen
  size_t bytes_read = 0;
  i2s_read(I2S_NUM_0, adc_buffer, sizeof(adc_buffer), &bytes_read, portMAX_DELAY);

  int samples = bytes_read / sizeof(uint16_t);

  Serial.println("<DATA_START>");
  for (int i = 0; i < samples; i++) {
    uint16_t raw = adc_buffer[i] & 0xFFF;
    uint32_t mv = esp_adc_cal_raw_to_voltage(raw, adc_chars);
    Serial.print(mv);
    if (i < samples - 1) Serial.print(",");
  }
  Serial.println();
  Serial.println("<DATA_END>");
}
#endif // EINDE ROLE_MASTER


// ====================================================================
// =================== CODE VOOR DE SLAVE =============================
// ====================================================================
#ifdef ROLE_SLAVE

#define SAMPLES_PER_WAVE 50

// Buffers voor de puls en voor de stilte
uint16_t pulse_buffer[SAMPLES_PER_WAVE];
const int SILENCE_BUFFER_SAMPLES = 64;
uint16_t silence_buffer[SILENCE_BUFFER_SAMPLES];

// Een 'volatile' vlag die door de interrupt wordt gezet
volatile bool triggerPulse = false;

// De Interrupt Service Routine (ISR)
void IRAM_ATTR pulse_trigger_isr() {
  triggerPulse = true;
}

void setup_dac() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_DAC_BUILT_IN),
    .sample_rate = 40000 * SAMPLES_PER_WAVE, // 2 MHz
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_MSB,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, NULL);
  i2s_set_dac_mode(I2S_DAC_CHANNEL_LEFT_EN);
}

void generate_one_cycle() {
  for (int i = 0; i < SAMPLES_PER_WAVE; i++) {
    uint8_t val = (uint8_t)(127.5 + 85 * sinf(2.0f * PI * i / SAMPLES_PER_WAVE));
    pulse_buffer[i] = val << 8;
  }
}

void fill_silence_buffer() {
  for (int i = 0; i < SILENCE_BUFFER_SAMPLES; i++) {
    silence_buffer[i] = 127 << 8; // 1.65V DC offset
  }
}

void setup() {
  // FIX: Match de monitor_speed uit je platformio.ini
  Serial.begin(2000000);

  generate_one_cycle();
  fill_silence_buffer();

  setup_dac();
  i2s_zero_dma_buffer(I2S_NUM_0);

  // FIX: Gebruik een interrupt voor betrouwbare trigger detectie
  pinMode(TRIGGER_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(TRIGGER_PIN), pulse_trigger_isr, RISING);

  Serial.println("SLAVE ready and waiting for trigger...");
}

void loop() {
  size_t bytes_written = 0;

  // FIX: Houd de DAC continu actief voor een schoon signaal
  if (triggerPulse) {
    // Stuur de puls als de interrupt is afgegaan
    i2s_write(I2S_NUM_0, pulse_buffer, sizeof(pulse_buffer), &bytes_written, portMAX_DELAY);
    triggerPulse = false; // Reset de vlag
  } else {
    // Stuur de rest van de tijd stilte om de DAC "warm" te houden
    i2s_write(I2S_NUM_0, silence_buffer, sizeof(silence_buffer), &bytes_written, portMAX_DELAY);
  }
}
#endif // EINDE ROLE_SLAVE