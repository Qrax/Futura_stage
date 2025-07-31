#include <Arduino.h>
#include "driver/i2s.h"
#include "driver/adc.h"

// --- Configuratie ---
#define ADC_PIN               36
#define ADC_CHANNEL           ADC1_CHANNEL_0 // GPIO 36 is ADC1_CH0
#define I2S_PORT              I2S_NUM_0

#define TARGET_SAMPLE_RATE    (400000)       // 400 kS/s, vraag
#define SAMPLES_PER_CHUNK     1024
#define NUM_CHUNKS            100            // totaal 102 400 samples

// hoeveel we willen printen
#define NUM_SAMPLES_TO_PRINT  50

uint16_t i2s_read_buffer[SAMPLES_PER_CHUNK];

void setup_adc_i2s() {
    i2s_config_t i2s_config = {
        .mode               = (i2s_mode_t)(I2S_MODE_MASTER|I2S_MODE_RX|I2S_MODE_ADC_BUILT_IN),
        .sample_rate        = TARGET_SAMPLE_RATE,
        .bits_per_sample    = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format     = I2S_CHANNEL_FMT_ONLY_RIGHT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags   = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count      = 8,
        .dma_buf_len        = SAMPLES_PER_CHUNK,
        .use_apll           = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk         = 0
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_11);
    i2s_set_adc_mode(ADC_UNIT_1, ADC_CHANNEL);

    Serial.println("ADC en I2S geconfigureerd.");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    setup_adc_i2s();
}

void loop() {
    Serial.println("--- Start nieuwe meting ---");

    size_t bytes_read;
    long total_samples = NUM_CHUNKS * SAMPLES_PER_CHUNK;
    long startTime    = micros();

    for(int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
        esp_err_t res = i2s_read(I2S_PORT,
                                                         i2s_read_buffer,
                                                         sizeof(i2s_read_buffer),
                                                         &bytes_read,
                                                         portMAX_DELAY);
        if(res != ESP_OK) {
            Serial.printf("Error reading chunk %d: %s\n", chunk, esp_err_to_name(res));
            return;
        }

        // print first 50 samples of chunk 0
        if(chunk == 0) {
            Serial.println("Eerste 50 samples van chunk 0:");
            for(int i = 0; i < NUM_SAMPLES_TO_PRINT; ++i) {
                Serial.printf("%3d: %u\n", i, i2s_read_buffer[i]);
            }
            Serial.println("--------------------------");
        }
    }

    long endTime = micros();
    long dur     = endTime - startTime;
    float rate   = (float)total_samples / dur * 1e6;

    Serial.printf("Tijd: %ld µs voor %ld samples -> %.2f S/s\n\n",
                                dur, total_samples, rate);
    delay(3000);
}
