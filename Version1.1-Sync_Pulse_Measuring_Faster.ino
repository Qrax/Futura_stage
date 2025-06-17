// === Includes en Globale Definities ===
#include <TUSS44x0_Ultrasonic.h>
#include <SPI.h>
#include <Energia.h>
#include <msp430.h>

#define SYNC_PIN 12            // Sync communicatie (M->S interrupt trigger)
#define DEBUG_PIN 11           // Optionele debug pin

#define NUM_SAMPLES  3072
uint16_t voutBuf2[NUM_SAMPLES];

tuss44x0 tuss;
uint32_t baudRate = 115200;
bool isMasterMode = true;
// Record Length:
// - Voor Master SYNC/MEASURE: Hoe lang meet de Master?
// - Voor Slave PULSE/Delayed: Hoe lang meet de Slave na zijn puls?
byte recordLength = 20;       // Default 100ms, aanpasbaar met L


// --- State Variabelen ---
// Master: Geen timing vlaggen meer nodig voor SYNC
// Slave
volatile bool slaveAwaitingDelayedPulse = false;
volatile unsigned long syncPulseReceivedTimeSlave = 0; // ISR zet dit (micros)
volatile unsigned long delayedPulseSendTimeSlave = 0;  // Doel tijdstip puls (micros)
volatile bool slaveSyncInterruptOccurred = false;      // ISR vlag
volatile unsigned long interruptCounter = 0;       // Debug teller

// === Interrupt Service Routine (ISR) voor Slave Sync Pin (Pin 12) ===
void slaveSyncPinISR() {
  interruptCounter++;
  if (!slaveAwaitingDelayedPulse) {
    syncPulseReceivedTimeSlave = micros(); // Tijd vastleggen
    slaveSyncInterruptOccurred = true;     // Vlag voor hoofdloop
  }
}

// === Helper Functie: printADCData ===
//void printADCData() { /* ... onveranderd ... */ }

// === setup() Functie ===
void setup() {
  Serial.begin(baudRate); while (!Serial);
  Serial.println("--- Master(DirectRec) / Slave(DelayedPulse 40kHz) ---");
  Serial.print("Default Record Length (Mstr/Slv): "); Serial.print(recordLength); Serial.println(" ms");
  pinMode(DEBUG_PIN, OUTPUT); digitalWrite(DEBUG_PIN, LOW);
  Serial.println("Debug Pin 11 configured.");
  analogReadResolution(12);
  tuss.initTUSS44x0BP(baudRate);
  tuss.captureAndPrintCheck(false);
  tuss.tuss44x0_regWrite(0x10, 0x00); tuss.tuss44x0_regWrite(0x16, 0x00); tuss.tuss44x0_regWrite(0x1A, 0x08);

  // ADC op A5
  P6SEL |= BIT5;
  ADC12CTL0 = ADC12ON | ADC12SHT0_0;
  ADC12CTL1 = ADC12SHP;
  ADC12CTL2 = ADC12RES_2;
  ADC12MCTL0 = ADC12INCH_5;
  ADC12CTL0 |= ADC12ENC;

  slaveAwaitingDelayedPulse = false; slaveSyncInterruptOccurred = false; interruptCounter = 0;
  pinMode(SYNC_PIN, OUTPUT); digitalWrite(SYNC_PIN, LOW); // Start Master
  Serial.println("Mode set to MASTER (Pin 12 OUT, INT OFF).");

  Serial.println("Commands:");
  Serial.println("  M      - Set MASTER");
  Serial.println("  S      - Set SLAVE");
  Serial.println("  SYNC   - (Master: HW Sync->Direct ADC Rec | Slave: INT->50ms wait->Pulse->ADC Rec)"); // NIEUW
  Serial.println("  MEASURE- (Master: Record ADC now, no burst/sync)"); // NIEUW
  Serial.println("  PULSE  - (Slave: Send DIRECT 40kHz pulse & measure now)"); // (Was er al)
  Serial.println("  P      - Print SYNC_PIN (Pin 12) state");
  Serial.println("  L <ms> - Set Default ADC record length (e.g., L 100)"); // (Was er al)
}

// === Helper Functie: Stuur Debug Puls ===
void sendDebugPulse() {
  digitalWrite(DEBUG_PIN, HIGH); delayMicroseconds(10); digitalWrite(DEBUG_PIN, LOW);
}

// === loop() Functie ===
void loop() {
  // --- Verwerk Seriële Commando's ---
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n'); cmd.trim(); cmd.toUpperCase();

    if (cmd.equals("M")) { // Set Master
      isMasterMode = true; detachInterrupt(digitalPinToInterrupt(SYNC_PIN));
      slaveAwaitingDelayedPulse = false; slaveSyncInterruptOccurred = false; interruptCounter = 0;
      pinMode(SYNC_PIN, OUTPUT); digitalWrite(SYNC_PIN, LOW); pinMode(DEBUG_PIN, OUTPUT); digitalWrite(DEBUG_PIN, LOW);
      Serial.println("Mode: MASTER");
    } else if (cmd.equals("S")) { // Set Slave
      isMasterMode = false;
      slaveAwaitingDelayedPulse = false; slaveSyncInterruptOccurred = false; interruptCounter = 0;
      pinMode(SYNC_PIN, INPUT_PULLDOWN); pinMode(DEBUG_PIN, OUTPUT); digitalWrite(DEBUG_PIN, LOW);
      attachInterrupt(digitalPinToInterrupt(SYNC_PIN), slaveSyncPinISR, RISING);
      Serial.println("Mode: SLAVE");
    }
    // --- SYNC: Master meet direct, Slave start timer ---
    else if (cmd.equals("SYNC")) {
      if (isMasterMode) {
        //Serial.println("Master: SYNC cmd -> HW pulse Pin 12 (5us) & Start ADC Record...");
        // Stuur hardware sync puls
        tuss.triggerSyncPulse(); // Aanname: 5us in library!

        // Start DIRECT de meting (Record Only)
        //Serial.println("Master: Sending DEBUG PULSE on Pin 11 (Start Measure)...");
        sendDebugPulse(); // Debug Puls voor start meting
        tuss.tuss44x0_regWrite(0x1B, 0x00); tuss.tuss44x0_regWrite(0x1B, 0x01);
        //tuss.captureADC(recordLength); // Gebruik default/L lengte

        MeasureADCFast();

        tuss.tuss44x0_regWrite(0x1B, 0x00);
        //Serial.println("Master: SYNC Recording complete.");
        //printADCData(); // Stuur data
      } //else { Serial.println("Slave: Ignoring SYNC command (wacht op INT)."); }
    }
    // --- MEASURE: Master Meet Nu (Record Only) ---
    else if (cmd.equals("MEASURE")) {
      if (isMasterMode) {
        Serial.println("Master: MEASURE cmd - Recording ADC data now (no burst)...");
        Serial.println("Master: Sending DEBUG PULSE on Pin 11 (Start Measure)...");
        sendDebugPulse(); // Debug Puls
        tuss.tuss44x0_regWrite(0x1B, 0x00); tuss.tuss44x0_regWrite(0x1B, 0x01);
        //tuss.captureADC(recordLength); // Gebruik default/L lengte

        MeasureADCFast();

        tuss.tuss44x0_regWrite(0x1B, 0x00);
        Serial.println("Master: MEASURE Recording complete.");
        //printADCData();
      } else {
        Serial.println("Slave: MEASURE command not used.");
      }
    }
    // --- PULSE: Slave Pulse & Measure Now ---
    else if (cmd.equals("PULSE")) {
      if (!isMasterMode) {
        Serial.println("Slave: PULSE cmd - DIRECT 40kHz pulse & measure...");
        Serial.println("Slave: Sending DEBUG PULSE on Pin 11 (Start Pulse)...");
        sendDebugPulse(); // Debug puls
        tuss.tuss44x0_regWrite(0x1B, 0x00); tuss.tuss44x0_regWrite(0x1B, 0x01);
        noInterrupts(); tuss.burstBitBang(25, 1); interrupts(); // 40kHz puls
        tuss.tuss44x0_regWrite(0x1B, 0x00);
        // 1) Pak het start‑tijdstip

        MeasureADCFast();


        //tuss.captureADC(recordLength); // Meting direct erna
        Serial.println("Slave: PULSE measurement complete.");
        //printADCData();
      } else {
        Serial.println("PULSE command only valid in SLAVE mode.");
      }
    }
    // --- P: Check Pin ---
    else if (cmd.equals("P")) {
      Serial.println("Reading SYNC_PIN (Pin 12) state 30 times:");
      pinMode(SYNC_PIN, INPUT); delay(1);
      for (int i = 0; i < 30; i++) {
        int state = digitalRead(SYNC_PIN);
        Serial.print("M");
        Serial.print(i + 1);
        Serial.print("(P12): ");
        Serial.println(state == HIGH ? "H" : "L");
        delay(100);
      }
      if (isMasterMode) {
        pinMode(SYNC_PIN, OUTPUT);
        digitalWrite(SYNC_PIN, LOW);
      }
      else {
        pinMode(SYNC_PIN, INPUT_PULLDOWN);
        attachInterrupt(digitalPinToInterrupt(SYNC_PIN), slaveSyncPinISR, RISING);
      }
      Serial.println("P Measurement complete.");
    }
    // --- L: Set Default Record Length ---
    else if (cmd.startsWith("L ")) {
      String valueStr = cmd.substring(2); valueStr.trim();
      int newLength = valueStr.toInt();
      if (newLength > 0 && newLength <= 255) {
        recordLength = (byte)newLength;
        Serial.print("Default Record length set to: ");
        Serial.print(recordLength);
        Serial.println(" ms");
      }
      else {
        Serial.println("Invalid length (1-255 ms).");
      }
    }
    // --- Onbekend ---
    else {
      Serial.println("Unknown command.");
    }
  } // Einde Serial

  // --- GEEN Master Delayed Measurement Logic meer nodig ---

  // --- Automatische Slave Delayed Reactie (50ms delay) ---
  if (!isMasterMode) {
    bool interruptFlag;
    noInterrupts(); interruptFlag = slaveSyncInterruptOccurred; interrupts();

    // Als interrupt net gebeurd is, start de timer
    if (interruptFlag && !slaveAwaitingDelayedPulse) {
      unsigned long capturedTime;
      noInterrupts(); capturedTime = syncPulseReceivedTimeSlave; slaveSyncInterruptOccurred = false; interrupts(); // Reset vlag!
      // Bereken doel tijdstip: ontvangst + 5 milliseconden (in microseconden)
      delayedPulseSendTimeSlave = capturedTime + 1500UL; // <<< 3ms delay
      slaveAwaitingDelayedPulse = true;
      //Serial.print("Slave: INT Pin 12! Time(us): "); Serial.println(capturedTime);
      //Serial.println("Slave: Waiting 50ms for pulse & measurement..."); // Aangepast bericht
    }

    // Als we wachten en de tijd is voorbij
    if (slaveAwaitingDelayedPulse && (micros() >= delayedPulseSendTimeSlave)) {
      //Serial.println("Slave: 50ms voorbij -> Pulsing(40kHz) & Measuring..."); // Aangepast bericht
      //Serial.println("Slave: Sending DEBUG PULSE on Pin 11 (Start Pulse)...");
      sendDebugPulse(); // Debug puls Slave

      tuss.tuss44x0_regWrite(0x1B, 0x00); tuss.tuss44x0_regWrite(0x1B, 0x01);
      noInterrupts(); tuss.burstBitBang(25, 1); interrupts(); // 40kHz puls
      //Serial.println("Slave: Delayed 40kHz Pulse Sent.");
      tuss.tuss44x0_regWrite(0x1B, 0x00);


      MeasureADCFast();
      //tuss.captureADC(recordLength); // Meting na de puls

      //Serial.println("Slave: Delayed Measurement Complete.");
      //printADCData();
      slaveAwaitingDelayedPulse = false; // Reset vlag
    }

    // Optionele Interrupt Counter Print
    /* ... */
  } // Einde Slave Logic

  //delay(5);
}


void MeasureADCFast() {

  unsigned long startMicros = micros();

  // 2) Blocking sample‑lus met debug‑toggle en timestamp
  for (int i = 0; i < NUM_SAMPLES; i++) {           // toggle voor elke conversie
    ADC12CTL0 |= ADC12SC;
    while (!(ADC12IFG & ADC12IFG0));
    voutBuf2[i] = ADC12MEM0;
    ADC12IFG &= ~ADC12IFG0;
  }

  // 3) Pak het eind‑tijdstip
  unsigned long endMicros = micros();

  // 4) Bereken totaal‑duur en prints
  unsigned long totalTime = endMicros - startMicros;
  Serial.print("Total time from first to last sample: ");
  Serial.print(totalTime);
  Serial.println(" us");

  Serial.print("Number of samples: ");
  Serial.println(NUM_SAMPLES);

  Serial.print("Average per sample: ");
  Serial.print((float)totalTime / NUM_SAMPLES, 2);
  Serial.println(" us");

  // 5) Dump eerste 10 waarden
  Serial.println("Samples:");
  for (int i = 0; i < NUM_SAMPLES; i++) {
    Serial.println(voutBuf2[i]);
  }
}



// === Helper Functie: printADCData ===
void printADCData() {
  Serial.print("Captured Samples: "); Serial.println(capturedSamples);
  //unsigned long totalDiff = 0; // Oude manier

  // --- Gebruik float voor gemiddelde berekening ---
  if (capturedSamples > 1) {
    // We moeten totalDiff nog steeds als unsigned long berekenen
    // omdat de individuele verschillen groot kunnen worden.
    unsigned long totalDiff = 0;
    for (int i = 1; i < capturedSamples; i++) {
      // Voorkom negatieve resultaten bij micros() wrap-around (zeer onwaarschijnlijk hier)
      if (timeStampBuf[i] >= timeStampBuf[i - 1]) {
        totalDiff += (timeStampBuf[i] - timeStampBuf[i - 1]);
      } else {
        // Handle wrap-around if necessary, though unlikely for short captures
        totalDiff += ( (0xFFFFFFFF - timeStampBuf[i - 1]) + timeStampBuf[i] );
      }
    }

    // Bereken het gemiddelde als float
    float avgDiffFloat = (float)totalDiff / (float)(capturedSamples - 1); // Cast naar float!

    Serial.print("Average time between samples (us): ");
    Serial.print(avgDiffFloat, 1); // Print de float met 1 decimaal
    Serial.println(); // Voeg zelf de newline toe
  }
  // --- Einde float berekening ---

  else if (capturedSamples == 1) Serial.println("Only one sample captured.");
  else Serial.println("No samples captured.");

  Serial.println("Timestamp (us), ADC Value:");
  for (int i = 0; i < capturedSamples; i++) {
    Serial.print(timeStampBuf[i]); Serial.print(", "); Serial.println((uint16_t)voutBuf2[i]);
  }
  Serial.println("E");
}
