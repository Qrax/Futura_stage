#include <Arduino.h>
#include "soc/rtc_io_reg.h"
#include "soc/rtc_cntl_reg.h"
#include "soc/sens_reg.h"
#include "driver/dac.h"

// --- Configuratie ---
// De frequentie die we willen benaderen.
const int SINE_FREQUENCY_HZ = 40000;

// Aantal golven dat we per keer willen uitzenden.
const int WAVES_PER_BURST = 10;

// De pauze tussen de bursts in milliseconden.
const int PAUSE_DURATION_MS = 3000;

// Bereken de duur van de burst in microseconden.
// Duur = (1 / frequentie) * aantal golven * 1.000.000 (om naar microseconden te gaan)
const int BURST_DURATION_US = (1000000 / SINE_FREQUENCY_HZ) * WAVES_PER_BURST;


void setup() {
    // --- Configureer de Cosine Waveform (CW) generator ---
    // Dit hoeft maar één keer te gebeuren. De hardware blijft de golf intern
    // genereren, wij schakelen alleen de output naar de pin aan en uit.

    // 1. Schakel de CW generator in.
    SET_PERI_REG_MASK(SENS_SAR_DAC_CTRL1_REG, SENS_SW_TONE_EN);

    // 2. Verbind de CW generator met DAC kanaal 1 (GPIO 25).
    SET_PERI_REG_MASK(SENS_SAR_DAC_CTRL2_REG, SENS_DAC_CW_EN1_M);

    // 3. Stel de frequentie in.
    // Formule: freq_step = freq_hz * 65536 / 8_000_000
    uint16_t frequency_step = (uint64_t)SINE_FREQUENCY_HZ * 65536 / 8000000;
    SET_PERI_REG_BITS(SENS_SAR_DAC_CTRL1_REG, SENS_SW_FSTEP, frequency_step, SENS_SW_FSTEP_S);

    // 4. Corrigeer de golfvorm-inversie (standaard is geïnverteerd). '2' (10b) corrigeert dit.
    SET_PERI_REG_BITS(SENS_SAR_DAC_CTRL2_REG, SENS_DAC_INV1, 2, SENS_DAC_INV1_S);

    // BELANGRIJK: Zorg ervoor dat de DAC-output standaard uit staat.
    // We schakelen deze alleen in de loop in wanneer we een burst willen.
    dac_output_disable(DAC_CHANNEL_1);
}

void loop() {
    // --- Genereer een burst van 10 golven ---
    
    // 1. Schakel de output op DAC pin 25 in.
    dac_output_enable(DAC_CHANNEL_1);

    // 2. Wacht de berekende duur van de burst (250 microseconden).
    delayMicroseconds(BURST_DURATION_US);

    // 3. Schakel de output weer uit.
    dac_output_disable(DAC_CHANNEL_1);

    // 4. Wacht 3 seconden tot de volgende burst.
    delay(PAUSE_DURATION_MS);
}