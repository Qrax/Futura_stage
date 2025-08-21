// ==========================================================================
// Universal Time-Reversal Code - VOLLEDIGE CYCLUS (VERSIE 1.1 - BUGS GEFIXED)
// ==========================================================================

#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_adc_cal.h"

// --- Pin Configuratie (vast voor beide boards) ---
#define PIN_CMD 27    // Draad 1: Commando's (Master -> Slave)
#define PIN_STATUS 33 // Draad 2: Status (Slave -> Master)
#define PIN_CMD_IN PIN_CMD
#define PIN_CMD_OUT PIN_CMD
#define PIN_STATUS_IN PIN_STATUS
#define PIN_STATUS_OUT PIN_STATUS

// --- I2S & Puls Configuratie ---
#define I2S_PORT I2S_NUM_0
#define TARGET_SAMPLE_RATE 2000000
#define SINE_WAVE_FREQ_HZ 40000
#define PULSE_BUFFER_SAMPLES (TARGET_SAMPLE_RATE / SINE_WAVE_FREQ_HZ)
#define RECEIVE_BUFFER_SAMPLES 8192

// --- Globale Variabelen ---
bool isMaster = false;
volatile bool trigger_received = false;

enum MasterState { M_IDLE, M_WAITING_FOR_SLAVE, M_WAITING_FOR_USER_SWAP, M_LISTENING, M_DONE };
MasterState masterState = M_IDLE;

enum SlaveState { S_IDLE, S_RECEIVING, S_PROCESSING, S_READY_TO_SEND, S_SENDING };
SlaveState slaveState = S_IDLE;

// --- Buffers ---
uint16_t pulse_buffer[PULSE_BUFFER_SAMPLES];
uint16_t recorded_data[RECEIVE_BUFFER_SAMPLES];
uint16_t dac_buffer[RECEIVE_BUFFER_SAMPLES];
uint16_t final_echo_buffer[RECEIVE_BUFFER_SAMPLES];
esp_adc_cal_characteristics_t *adc_chars;

// --- Functie Prototypes ---
void setup_i2s_dac_mode();
void setup_i2s_adc_mode();
void generate_pulse_waveform();
void time_reverse_buffer(uint16_t* buffer, int length);
void convert_mv_to_dac(const uint16_t* mv_buffer, int length);

// --- Interrupt Service Routines ---
void IRAM_ATTR on_command_received_isr() { trigger_received = true; }
void IRAM_ATTR on_status_received_isr() { trigger_received = true; }

// ==========================================================================
// SETUP
// ==========================================================================
void setup() {
    Serial.begin(2000000);
    delay(3000);

    if (Serial) {
        isMaster = true;
    } else {
        isMaster = false;
        Serial.end();
    }

    if (isMaster) {
        pinMode(PIN_CMD_OUT, OUTPUT);
        digitalWrite(PIN_CMD_OUT, LOW);
        pinMode(PIN_STATUS_IN, INPUT_PULLDOWN);
        attachInterrupt(digitalPinToInterrupt(PIN_STATUS_IN), on_status_received_isr, RISING);
    } else { // isSlave
        pinMode(PIN_CMD_IN, INPUT_PULLDOWN);
        attachInterrupt(digitalPinToInterrupt(PIN_CMD_IN), on_command_received_isr, RISING);
        pinMode(PIN_STATUS_OUT, OUTPUT);
        digitalWrite(PIN_STATUS_OUT, LOW);
    }

    generate_pulse_waveform();
    adc1_config_width(ADC_WIDTH_BIT_12);
    // FIX: Gebruik de nieuwe, aanbevolen naam ADC_ATTEN_DB_12
    adc1_config_channel_atten(ADC1_CHANNEL_6, ADC_ATTEN_DB_12);
    adc_chars = (esp_adc_cal_characteristics_t *)calloc(1, sizeof(esp_adc_cal_characteristics_t));
    // FIX: Gebruik de nieuwe, aanbevolen naam ADC_ATTEN_DB_12
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_12, ADC_WIDTH_BIT_12, 1100, adc_chars);

    if (isMaster) {
        Serial.println("--- ROL: MASTER (Controller, aan PC) ---");
        Serial.println("Volledige Time-Reversal Cyclus");
        Serial.println("1. Verbind transducer op A met DAC (26), op B met ADC (34).");
        Serial.println("2. Druk op Enter om te starten.");
    }
}

// ==========================================================================
// LOOP
// ==========================================================================
void loop() {
    if (isMaster) {
        // --- MASTER LOGICA ---
        switch (masterState) {
            case M_IDLE:
                if (Serial.available() > 0) {
                    Serial.read();
                    Serial.println("\nStap 1: Initiële puls wordt verzonden...");
                    setup_i2s_dac_mode();
                    digitalWrite(PIN_CMD_OUT, HIGH);
                    delayMicroseconds(10);
                    size_t bytes_written;
                    i2s_write(I2S_PORT, pulse_buffer, sizeof(pulse_buffer), &bytes_written, portMAX_DELAY);
                    digitalWrite(PIN_CMD_OUT, LOW);
                    i2s_driver_uninstall(I2S_PORT);
                    Serial.println("Puls verzonden. Wachten op Slave...");
                    masterState = M_WAITING_FOR_SLAVE;
                }
                break;
            case M_WAITING_FOR_SLAVE:
                if (trigger_received) {
                    trigger_received = false;
                    Serial.println("Slave meldt: KLAAR!");
                    Serial.println("\n!!! ACTIE VEREIST !!!");
                    Serial.println("1. Verplaats kabel op A van DAC (26) naar ADC (34).");
                    Serial.println("2. Verplaats kabel op B van ADC (34) naar DAC (26).");
                    Serial.println("3. Druk op Enter om de echo-fase te starten.");
                    masterState = M_WAITING_FOR_USER_SWAP;
                }
                break;
            case M_WAITING_FOR_USER_SWAP:
                 if (Serial.available() > 0) {
                    Serial.read();
                    Serial.println("\nStap 2: Luisteren naar echo en Slave triggeren...");
                    setup_i2s_adc_mode();
                    digitalWrite(PIN_CMD_OUT, HIGH);
                    delayMicroseconds(10);
                    digitalWrite(PIN_CMD_OUT, LOW);
                    masterState = M_LISTENING;
                 }
                 break;
            case M_LISTENING: { // FIX: Accolade toegevoegd om scope-fout te voorkomen
                size_t bytes_read;
                i2s_read(I2S_PORT, final_echo_buffer, sizeof(final_echo_buffer), &bytes_read, portMAX_DELAY);
                i2s_driver_uninstall(I2S_PORT);
                Serial.println("Gefocuste echo ontvangen!");
                
                Serial.println("\n--- Eindresultaat: Gefocuste Echo ---");
                Serial.println("<DATA_START>");
                int samples_read = bytes_read / sizeof(uint16_t);
                for (int i = 0; i < samples_read; i++) {
                    uint16_t raw_value = final_echo_buffer[i] & 0xFFF;
                    uint32_t voltage_mv = esp_adc_cal_raw_to_voltage(raw_value, adc_chars);
                    Serial.print(voltage_mv);
                    if (i < samples_read - 1) Serial.print(",");
                }
                Serial.println();
                Serial.println("<DATA_END>");
                Serial.println("\n--- Volledige Cyclus Voltooid ---");
                masterState = M_DONE;
                break;
            } // FIX: Accolade toegevoegd
            case M_DONE: 
                break;
        }
    } 
    else {
        // --- SLAVE LOGICA ---
        if (trigger_received) {
            trigger_received = false;
            if (slaveState == S_IDLE) {
                slaveState = S_RECEIVING;
                setup_i2s_adc_mode();
                size_t bytes_read;
                i2s_read(I2S_PORT, recorded_data, sizeof(recorded_data), &bytes_read, portMAX_DELAY);
                i2s_driver_uninstall(I2S_PORT);
                
                slaveState = S_PROCESSING;
                int samples_read = bytes_read / sizeof(uint16_t);
                uint16_t mv_buffer[samples_read];
                for(int i=0; i<samples_read; i++){
                    uint16_t raw_value = recorded_data[i] & 0xFFF;
                    mv_buffer[i] = esp_adc_cal_raw_to_voltage(raw_value, adc_chars);
                }
                time_reverse_buffer(mv_buffer, samples_read);
                convert_mv_to_dac(mv_buffer, samples_read);
                
                slaveState = S_READY_TO_SEND;
                digitalWrite(PIN_STATUS_OUT, HIGH);
                delayMicroseconds(10);
                digitalWrite(PIN_STATUS_OUT, LOW);
            }
            else if (slaveState == S_READY_TO_SEND) {
                slaveState = S_SENDING;
                setup_i2s_dac_mode();
                size_t bytes_written;
                i2s_write(I2S_PORT, dac_buffer, sizeof(dac_buffer), &bytes_written, portMAX_DELAY);
                i2s_driver_uninstall(I2S_PORT);
                slaveState = S_IDLE;
            }
        }
    }
}

// ==========================================================================
// HELPER FUNCTIES
// ==========================================================================
void setup_i2s_dac_mode() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_DAC_BUILT_IN),
        .sample_rate = TARGET_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
        .communication_format = I2S_COMM_FORMAT_STAND_MSB,
        .intr_alloc_flags = 0,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = true,
        .tx_desc_auto_clear = true
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, NULL);
    i2s_set_dac_mode(I2S_DAC_CHANNEL_RIGHT_EN);
}

void setup_i2s_adc_mode() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
        .sample_rate = TARGET_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 1024,
        .use_apll = true,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_adc_mode(ADC_UNIT_1, ADC1_CHANNEL_6);
    i2s_adc_enable(I2S_PORT);
}

void generate_pulse_waveform() {
    for (int i = 0; i < PULSE_BUFFER_SAMPLES; i++) {
        uint8_t dac_value = (uint8_t)(127.5 + 127.0 * sinf(2.0f * PI * i / PULSE_BUFFER_SAMPLES));
        pulse_buffer[i] = dac_value << 8;
    }
}

void time_reverse_buffer(uint16_t* buffer, int length) {
    for (int i = 0; i < length / 2; i++) {
        uint16_t temp = buffer[i];
        buffer[i] = buffer[length - 1 - i];
        buffer[length - 1 - i] = temp;
    }
}

void convert_mv_to_dac(const uint16_t* mv_buffer, int length) {
    for(int i=0; i < length; i++) {
        uint32_t dac_val = (mv_buffer[i] * 255) / 3300;
        if (dac_val > 255) dac_val = 255;
        dac_buffer[i] = (uint16_t)dac_val << 8;
    }
}