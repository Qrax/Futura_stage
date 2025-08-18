// File: src/main_rx.cpp

#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_adc_cal.h" // Bibliotheek voor ADC kalibratie

// --- Configuratie ---
#define ADC_PIN_ENUM ADC1_CHANNEL_6 // ADC1_CHANNEL_6 is GPIO 34.
#define TARGET_SAMPLE_RATE 1000000 // Doel sample rate in Hz (1 MHz). Probeer 500000 als dit problemen geeft.

// Aantal samples per burst. Bepaalt de meetduur.
// Duur = 4096 / 1000000 = 4.096 ms
#define BUFFER_SAMPLES 4096

const unsigned long INTERVAL_MS = 5000; // 5 seconden tussen bursts

// --- Globale Variabelen ---
uint16_t adc_buffer[BUFFER_SAMPLES];
unsigned long lastSampleTime = 0;
esp_adc_cal_characteristics_t *adc_chars; // Pointer voor kalibratiedata

// Functie om de I2S & ADC hardware te configureren
void setup_adc_i2s() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
        .sample_rate = TARGET_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4, // Meer buffers voor stabiliteit op hoge snelheid
        .dma_buf_len = 1024,
        .use_apll = true, // Gebruik APLL voor hogere precisie klok
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_adc_mode(ADC_UNIT_1, ADC_PIN_ENUM);
    i2s_adc_enable(I2S_NUM_0);
}

// Functie om de I2S driver weer vrij te geven
void stop_adc_i2s() {
    i2s_adc_disable(I2S_NUM_0);
    i2s_driver_uninstall(I2S_NUM_0);
}

void setup() {
    Serial.begin(2000000); // Hogere baudrate voor snelle dataoverdracht
    delay(100);

    // --- ADC Kalibratie Setup ---
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC_PIN_ENUM, ADC_ATTEN_DB_11);
    
    // Haal de kalibratiekarakteristieken op voor de interne Vref
    adc_chars = (esp_adc_cal_characteristics_t *)calloc(1, sizeof(esp_adc_cal_characteristics_t));
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, adc_chars);

    Serial.println("High-Speed ADC Sampler met kalibratie is klaar.");
}

void loop() {
    if (millis() - lastSampleTime >= INTERVAL_MS) {
        lastSampleTime = millis();

        setup_adc_i2s();
        
        size_t bytes_read = 0;
        unsigned long startTime = micros();
        i2s_read(I2S_NUM_0, adc_buffer, sizeof(adc_buffer), &bytes_read, portMAX_DELAY);
        unsigned long durationMicros = micros() - startTime;
        
        stop_adc_i2s();

        int samples_read = bytes_read / sizeof(uint16_t);
        float effective_sample_rate = (float)samples_read / (durationMicros / 1000000.0);

        // Stuur metadata
        Serial.printf("Effective Sample Rate: %.2f Hz\n", effective_sample_rate);
        Serial.printf("Samples Read: %d\n", samples_read);

        // Stuur de data, omringd door markers
        Serial.println("<DATA_START>");
        for (int i = 0; i < samples_read; i++) {
            uint16_t raw_value = adc_buffer[i] & 0xFFF;
            // Converteer de ruwe waarde naar een gekalibreerde spanning in millivolt
            uint32_t voltage_mv = esp_adc_cal_raw_to_voltage(raw_value, adc_chars);
            Serial.print(voltage_mv);
            if (i < samples_read - 1) {
                Serial.print(",");
            }
        }
        Serial.println(); // Nieuwe regel na de laatste waarde
        Serial.println("<DATA_END>");
    }
}