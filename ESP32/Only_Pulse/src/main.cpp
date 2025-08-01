#include <Arduino.h>

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("--- Geheugenstatus bij opstarten ---");

    // ESP.getHeapSize(): De totale grootte van het dynamische geheugen (de Heap).
    Serial.printf("Totale Heap grootte: %u bytes\n", ESP.getHeapSize());

    // ESP.getFreeHeap(): Hoeveel van de Heap is NU vrij.
    Serial.printf("Vrije Heap NU: %u bytes\n", ESP.getFreeHeap());

    // ESP.getMinFreeHeap(): De kleinste hoeveelheid vrije Heap sinds de start.
    // DIT IS DE BELANGRIJKSTE WAARDE om crashes te voorkomen!
    // Het is het 'laagwaterpeil' van je geheugen.
    Serial.printf("Minimale vrije Heap (laagwaterpeil): %u bytes\n", ESP.getMinFreeHeap());
    Serial.println("-------------------------------------\n");
}

void loop() {
    // Laten we de status elke 5 seconden printen om te zien hoe het verandert.
    Serial.println("--- Geheugenstatus in loop ---");
    Serial.printf("Vrije Heap NU: %u bytes\n", ESP.getFreeHeap());
    Serial.printf("Minimale vrije Heap (laagwaterpeil): %u bytes\n", ESP.getMinFreeHeap());
    Serial.println("-----------------------------\n");
    
    // Stel je voor dat je hier een grote array zou aanmaken:
    // uint8_t* big_array = (uint8_t*) malloc(50000); // Vraag 50KB aan
    // if (big_array) {
    //    Serial.println("50KB succesvol gereserveerd!");
    //    free(big_array); // Geef het geheugen weer vrij
    // } else {
    //    Serial.println("Geheugen reserveren mislukt!");
    // }

    delay(5000);
}