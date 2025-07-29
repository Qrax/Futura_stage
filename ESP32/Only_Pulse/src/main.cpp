// AANPASSING 1: Deze regel is verplicht in PlatformIO
#include <Arduino.h>

// ==== CONSTANTEN VOOR DE PULS ====
const int PULSE_SAMPLE_COUNT = 20;
// Vooraf berekende 8-bit sinusgolf data (0-255, gecentreerd op 128)
const byte sinus40kHz[PULSE_SAMPLE_COUNT] = {
  128, 168, 205, 235, 252, 255, 243, 218, 183, 143,
  103, 67, 37, 18, 8, 22, 51, 85, 118, 128
};
// Timing voor 40kHz (25us per cyclus / 20 samples = 1.25us per sample. We gebruiken 1us)
const int PULSE_DELAY_US = 1;


// ==== PIN DEFINITIES & WACHTZONES ====
const int QUIET_ZONE_MS = 100;
const int DRV_EN_PIN = 23;
const int DRV_GAIN0_PIN = 22;
const int DRV_GAIN1_PIN = 21;
const int DAC_PIN = 25; // DAC1 is op GPIO25


// ++++ FUNCTIE OM PULSEN TE SPELEN ++++
// AANPASSING 2: Deze functie is naar boven verplaatst, zodat 'loop()' hem kan vinden.
void playPulse(int numPulses) {
  Serial.printf("Driver aanzetten en %d puls(en) versturen...\n", numPulses);

  // STAP 1: Zet de driver aan met een neutrale DAC output.
  dacWrite(DAC_PIN, 128); // Begin altijd op het nulpunt
  digitalWrite(DRV_EN_PIN, HIGH);
  delay(QUIET_ZONE_MS); // Wachtzone voor stabilisatie

  // STAP 2: Speel het aantal pulsen af
  for (int p = 0; p < numPulses; p++) {
    for (int i = 0; i < PULSE_SAMPLE_COUNT; i++) {
      dacWrite(DAC_PIN, sinus40kHz[i]);
      delayMicroseconds(PULSE_DELAY_US);
    }
  }

  // STAP 3: Keer terug naar neutraal en zet uit
  dacWrite(DAC_PIN, 128); // Eindig altijd op het nulpunt
  delay(QUIET_ZONE_MS); // Wachtzone voor stabilisatie
  digitalWrite(DRV_EN_PIN, LOW);

  Serial.println("...Klaar.");
}


void setup() {
  // --- ESSENTIËLE DEBUGGING STAP ---
  // Een kleine vertraging om de seriële monitor de tijd te geven om te verbinden na een (her)start.
  delay(1000);

  Serial.begin(115200);
  Serial.println("\n--- Verbeterde Puls & Waveform Generator Initialisatie ---");

  // --- OPLOSSING: EXPLICIETE PINMODUS INSTELLEN ---
  // Het is cruciaal om de modus van de DAC-pin expliciet in te stellen voordat je
  // dacWrite() gebruikt. Het niet instellen hiervan is een veelvoorkomende oorzaak van crashes.
  Serial.println("Configureren van GPIO pinnen...");
  pinMode(DAC_PIN, OUTPUT);
  pinMode(DRV_EN_PIN, OUTPUT);
  pinMode(DRV_GAIN0_PIN, OUTPUT);
  pinMode(DRV_GAIN1_PIN, OUTPUT);

  // Zet gain op de laagste stand
  digitalWrite(DRV_GAIN0_PIN, LOW);
  digitalWrite(DRV_GAIN1_PIN, LOW);

  Serial.println("Setup voltooid. Starten van de loop...");
}

void loop() {
  Serial.println("\nLoop gestart. De testsequentie wordt eenmalig uitgevoerd.");

  // --- Test 1: Stuur één enkele 40 kHz puls ---
  Serial.println("\n[TEST 1] Een enkele 40kHz puls wordt over 1 seconde verstuurd...");
  delay(1000);
  playPulse(10);

  // // --- Test 2: Stuur 5 pulsen achter elkaar ---
  // Serial.println("\n[TEST 2] Vijf pulsen worden over 3 seconden verstuurd...");
  // delay(3000);
  // playPulse(5);

  Serial.println("\n--- Einde van de testsequentie. De ESP32 zal nu niets meer doen. Herstart om opnieuw te testen. ---");
  
  // --- VERBETERING: VOORKOM HERHALING VAN DE LOOP ---
  // Dit voorkomt dat de loop onbedoeld opnieuw start, wat nuttig is voor debuggen.
  // De ESP32 blijft aan, maar voert geen nieuwe tests uit.
  while (true) {
    delay(10000); // Wacht 10 seconden om de CPU niet onnodig te belasten.
  }
}