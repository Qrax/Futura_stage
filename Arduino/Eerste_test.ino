#include "signal_data.h" // <-- BELANGRIJK: Hiermee laden we de data!

// ==== Pin Definities ====
const int DRV_EN_PIN = 23;
const int DRV_GAIN0_PIN = 22;
const int DRV_GAIN1_PIN = 21;
const int DAC_PIN = 25; // DAC_CHANNEL_1 op de meeste ESP32 boards
const int QUIET_ZONE_MS = 1000;

// De variabelen 'numSamples' en 'SAMPLE_DELAY_US' komen nu uit 'signal_data.h'

void setup() {
  Serial.begin(115200);
  while (!Serial); // Wacht eventueel tot de Serial Monitor open is
  
  Serial.println("--- TUSS Omgekeerd Signaal Generator ---");
  Serial.printf("Signaal geladen met %d samples.\n", numSamples);
  Serial.printf("Sample delay ingesteld op %d us.\n", SAMPLE_DELAY_US);
  
  pinMode(DRV_EN_PIN, OUTPUT);
  pinMode(DRV_GAIN0_PIN, OUTPUT);
  pinMode(DRV_GAIN1_PIN, OUTPUT);

  // Stel de gain in. Voorbeeld: 28.5 dB (zie DRV2700 datasheet)
  // Dit kun je aanpassen als het signaal te sterk of te zwak is.
  digitalWrite(DRV_GAIN0_PIN, LOW);
  digitalWrite(DRV_GAIN1_PIN, LOW);
}

void loop() {
  Serial.println("\n[1] Het omgekeerde signaal wordt over 1 seconden afgespeeld...");
  delay(1000);
  
  playWaveform(true); // true = speel in reverse af
  
  Serial.println("\nKlaar. Wacht 5 seconden voor de volgende ronde.");
  delay(4000);
}

// De meest robuuste playWaveform functie met geïsoleerde wachtzones
void playWaveform(bool reverse) {
  const byte neutralValue = 128;

  // --- STAP 1: OPSTARTEN & STABILISEREN ---
  Serial.printf("\n--- Start Cyclus ---\n");
  Serial.println("Driver wordt aangezet...");

  // Zorg dat de DAC neutraal is VOORDAT de driver aangaat
  dacWrite(DAC_PIN, neutralValue);
  
  // Zet de driver aan
  digitalWrite(DRV_EN_PIN, HIGH);
  
  // WACHTZONE 1: Geef de driver en de voeding RUIM de tijd om te stabiliseren.
  // De opstart-plop gebeurt hier.
  Serial.printf("Start %d ms Pre-Signaal Wachtzone...\n", QUIET_ZONE_MS);
  delay(QUIET_ZONE_MS);
  Serial.println("...Wachtzone voorbij.");


  // --- STAP 2: PUUR SIGNAAL AFSPELEN ---
  Serial.println("Signaal wordt afgespeeld...");
  
  // Bepaal de loop richting
  int startIdx = reverse ? numSamples - 1 : 0;
  int endIdx = reverse ? -1 : numSamples;
  int step = reverse ? -1 : 1;

  for (int i = startIdx; i != endIdx; i += step) {
    byte dacValue = pgm_read_byte_near(signalData + i);
    dacWrite(DAC_PIN, dacValue);
    delayMicroseconds(SAMPLE_DELAY_US);
  }
  Serial.println("...Signaal afgespeeld.");


  // --- STAP 3: AFSLUITEN & STABILISEREN ---
  // WACHTZONE 2: Houd de laatste DAC-waarde vast en wacht.
  // Het signaal is nu voorbij, maar de driver staat nog aan.
  Serial.printf("Start %d ms Post-Signaal Wachtzone...\n", QUIET_ZONE_MS);
  delay(QUIET_ZONE_MS);
  Serial.println("...Wachtzone voorbij.");

  // Zet de driver uit. De afsluit-plop gebeurt hier.
  digitalWrite(DRV_EN_PIN, LOW);
  Serial.println("Driver is UIT.");
  
  // Kleine extra delay om zeker te zijn dat de driver echt "doof" is.
  delay(50);
  
  // Reset de DAC naar neutraal voor de volgende ronde.
  dacWrite(DAC_PIN, neutralValue);
  Serial.println("--- Einde Cyclus ---");
}