# Van Stabiele Vingerafdruk naar Metastabiliteit: Ultrasone Analyse van Defectdiepte in Aluminium en G10

Dit repository bevat de volledige software die is ontwikkeld voor de afstudeerscriptie met dezelfde naam, geschreven door Quincy Koelman voor de opleiding Technische Natuurkunde aan de Hogeschool van Amsterdam (Juni 2025).

## Overzicht

Dit project is een haalbaarheidsstudie naar een laagdrempelige, ultrasone non-destructief onderzoek (NDO) methode voor de inspectie van composietmaterialen, met name relevant voor de MRI-industrie. De kern van het onderzoek is niet de signaalverzwakking, maar de analyse van de **vorm van de signaal-envelop** om een unieke 'vingerafdruk' van een defect te identificeren.

De methode is getest op aluminium (6082) en G10-composiet, waarin modeldefecten (U-vormige gleuven van 5 mm en 15 mm) zijn aangebracht.

De software is opgedeeld in drie hoofdonderdelen:
1.  **Firmware** voor de MSP430 microcontrollers.
2.  Een **Python-script voor data-acquisitie** om de metingen aan te sturen en data op te slaan.
3.  Een set **Python-scripts voor data-analyse en visualisatie** om de resultaten te verwerken en de grafieken uit de scriptie te genereren.

## Repository Structuur

De code is georganiseerd in drie hoofdmappen:

-   `Energia Firmware/`: Bevat de C++ broncode voor de MSP430-microcontrollers die de ultrasone modules aansturen.
    -   Het `.ino` hoofdbestand regelt de meetcyclus en de communicatie.
    -   Bevat een aangepaste bibliotheek om de synchronisatie tussen de twee ultrasone modules (Master en Slave) via een GPIO-pin mogelijk te maken.

-   `Python - Data-acquisitie/`: Bevat het Python-script om de metingen vanaf een PC uit te voeren.
    -   Zet de seriële communicatie op met de modules.
    -   Stuurt de Master-module aan om een meetcyclus te starten.
    -   Ontvangt de meetdata (ADC-waarden van de signaal-envelop) en slaat deze op in `.csv`-bestanden voor latere analyse.

-   `Python - Analyse en Visualisatie/`: Bevat de Python-scripts die gebruikt zijn om de opgeslagen data te analyseren en de grafieken uit de scriptie te genereren.
    -   Een `main.py` script roept de verschillende sub-scripts aan.
    -   Scripts voor het middelen en uitlijnen van signalen (zie Bijlage E).
    -   Scripts voor het toepassen van detrending en het uitvoeren van een Fast Fourier Transform (FFT) om een periodogram te genereren.
    -   Scripts voor het visualiseren van de signaal-enveloppen en periodogrammen, zoals gepresenteerd in Hoofdstuk 4.

## Hardware

De experimentele opstelling is gebaseerd op een 'System-on-Chip' (Texas Instruments TUSS4440) en is gedetailleerd beschreven in de scriptie. De belangrijkste componenten zijn:
-   **Microcontroller**: 2x TI MSP-EXP430F5529LP LaunchPad
-   **Ultrasone Front-End**: 2x TI BOOSTXL-TUSS4440 BoosterPack
-   **Transducers**: 2x PUI Audio UTR-1440K-TT-R (40 kHz)
-   **Mechanica**: Een op maat gemaakte, 3D-geprinte testhouder (zie Bijlage C).
-   **Testobjecten**: Aluminium (6082) en G10-composiet blokken (100x20x20 mm), zowel defectloos als met gefreesde gleuven van 5 mm en 15 mm diep.

Een volledige lijst van materialen is te vinden in Bijlage A van de scriptie.

## Gebruik

Volg deze stappen om de experimenten te reproduceren:

### 1. Hardware Setup
1.  Assembleer de ultrasone modules (Transducer + BoosterPack + LaunchPad) zoals getoond in Figuur 3 van de scriptie.
2.  Verbind de Master- en Slave-modules via een Dupont-draad tussen de aangewezen GPIO-pinnen voor synchronisatie.
3.  Verbind beide LaunchPads via USB met een PC.

### 2. Firmware
1.  Open de code in de `Energia Firmware/`-map met de [Energia IDE](http://energia.nu/).
2.  Flash de firmware naar beide MSP430-microcontrollers.

### 3. Data Acquisitie
1.  Navigeer naar de `Python - Data-acquisitie/` map.
2.  Installeer de benodigde Python-libraries (`pyserial`, `pandas`, `numpy`).
3.  Plaats een testobject in de testhouder en breng koppelingsgel aan.
4.  Voer het Python-script uit. Het script zal automatisch de seriële poorten detecteren en de rollen (Master/Slave) toewijzen.
5.  Volg de instructies in de terminal om een meet-run te starten. De data wordt automatisch opgeslagen in een CSV-bestand.

### 4. Data Analyse en Visualisatie
1.  Navigeer naar de `Python - Analyse en Visualisatie/` map.
2.  Zorg dat de benodigde libraries (`pandas`, `numpy`, `matplotlib`, `scipy`) geïnstalleerd zijn.
3.  Plaats de gegenereerde `.csv`-bestanden in de daarvoor bestemde data-map.
4.  Voer het `main.py` script uit om de analyses uit te voeren en de figuren te genereren. De gedetailleerde stappen van de dataverwerking (uitlijnen, middelen, detrenden, FFT) zijn beschreven in Bijlage E van de scriptie.

## Auteur
**Quincy Koelman**
-   [GitHub Profiel](https://github.com/Qrax)
