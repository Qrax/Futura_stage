#include <Arduino.h>
#include "driver/i2s.h"
#include "driver/adc.h"
#include "esp_heap_caps.h"

// --- CONFIGURATIE ---
#define ADC_PIN               36
#define ADC_CHANNEL           ADC1_CHANNEL_0
#define I2S_PORT              I2S_NUM_0
#define SAMPLE_RATE_HZ        (400000)
#define TOTAL_SAMPLES         4096

// De grote buffer om alle data in op te slaan
uint16_t* measurement_buffer = NULL;

void setup() {
    Serial.begin(921600);
    delay(1000);

    // --- SETUP I2S en ADC ---
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
        .sample_rate = SAMPLE_RATE_HZ,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_11);
    i2s_set_adc_mode(ADC_UNIT_1, ADC_CHANNEL);
    Serial.println("Hardware drivers geconfigureerd.");

    // Geheugen reserveren
    size_t buffer_size = TOTAL_SAMPLES * sizeof(uint16_t);
    measurement_buffer = (uint16_t*) heap_caps_malloc(buffer_size, MALLOC_CAP_DMA);
    if (measurement_buffer == NULL) {
        Serial.println("FATALE FOUT: Geheugen reserveren mislukt!");
        while(1) {}
    }
    Serial.printf("Buffer van %u bytes gereserveerd.\n", buffer_size);
    Serial.println("Instrument klaar. Stuur 'm' om een meting te starten.");
}

void perform_measurement_and_transfer() {
    Serial.println("Commando ontvangen. Start meting...");

    // STAP 1: Reset de DMA buffer voor een schone start.
    i2s_zero_dma_buffer(I2S_PORT);
    
    // STAP 2: Start de ADC. De DMA begint nu op de achtergrond de buffers te vullen.
    i2s_adc_enable(I2S_PORT);
    
    // STAP 3: Start de tijdmeting.
    long startTime = micros();

    // STAP 4: Lees de data chunk voor chunk. Deze aanroep WACHT nu wel, omdat we
    // de meting net pas gestart zijn en de DMA-buffers nog niet vol zijn.
    size_t total_bytes_read = 0;
    while(total_bytes_read < TOTAL_SAMPLES * sizeof(uint16_t)) {
        size_t bytes_read_now = 0;
        // Wacht maximaal 100ms per chunk (fail-safe)
        i2s_read(I2S_PORT, (char*)measurement_buffer + total_bytes_read, 
                 TOTAL_SAMPLES * sizeof(uint16_t) - total_bytes_read, 
                 &bytes_read_now, 100);
        total_bytes_read += bytes_read_now;
    }

    // STAP 5: Stop de tijdmeting.
    long endTime = micros();
    
    // STAP 6: Stop de ADC.
    i2s_adc_disable(I2S_PORT);
    
    Serial.println("Meting voltooid. Start overdracht...");

    // --- FASE 2: VERSTUREN ---
    uint32_t samples_read = total_bytes_read / sizeof(uint16_t);
    for (int i = 0; i < samples_read; i++) {
        Serial.println(measurement_buffer[i]);
    }

    // --- FASE 3: METADATA ---
    long duration = endTime - startTime;
    float rate = 0.0;
    if (duration > 0 && samples_read > 0) {
        rate = (float)samples_read / duration * 1e6;
    }
    
    Serial.printf("DEBUG_DURATION:%ld\n", duration);
    Serial.printf("DEBUG_SAMPLES_READ:%d\n", samples_read);
    Serial.printf("RATE:%.2f\n", rate); 
    Serial.println("END"); 
}


void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'm') {
      perform_measurement_and_transfer();
    }
  }
}