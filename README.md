# Software voor Ultrasone Defect Analyse

Dit repository bevat de software die is ontwikkeld ter ondersteuning van een afstudeerproject voor de opleiding Technische Natuurkunde aan de Hogeschool van Amsterdam. Het project richt zich op een haalbaarheidsstudie naar niet-destructief onderzoek (NDO) met behulp van ultrasone analyse.

Het systeem is opgebouwd rond twee gesynchroniseerde Texas Instruments TUSS4440 ultrasone front-ends, die elk worden aangestuurd door een MSP-EXP430F5529LP microcontroller. De software is verdeeld in twee hoofdonderdelen: de firmware voor de microcontrollers en een verzameling Python-scripts voor data-acquisitie en -analyse.

## Repository Structuur

De code is georganiseerd in twee hoofddirectories:

-   **`Energia/`**: Bevat de firmware voor de MSP430 microcontrollers.
    -   `TUSS_Master_Slave.ino` (voorbeeldnaam): Het hoofdbestand voor de Energia IDE. Deze code bestuurt de TUSS4440 voor het genereren van pulsen en het ontvangen van data. Het bevat de logica om als "Master" of "Slave" te functioneren voor gesynchroniseerde metingen.
    -   Aangepaste library-bestanden: De twee meegeleverde library-bestanden zijn aangepast om de specifieke synchronisatiemethode van dit project mogelijk te maken.

-   **`Python/`**: Bevat alle Python-scripts voor het aansturen van het experiment en het analyseren van de data.
    -   `meerdere_metingen.py`: Het primaire script voor data-acquisitie. Het communiceert met de twee microcontrollers via seriële (COM) poorten, start de meetrondes, en slaat de resulterende data op in `.csv`-bestanden.
    -   `plotting/`: Een sub-directory met alle scripts die gerelateerd zijn aan data-analyse en visualisatie.
        -   `plotting_master.py`: Het **centrale controlescript** voor alle analyses. Dit bestand pas je aan om te selecteren welke databestanden je wilt laden en welke analyse-plots je wilt genereren.
        -   `plot_modes/`: Een directory met modulaire Python-scripts. Elk script hierin correspondeert met een specifieke "plot mode" die aangeroepen kan worden door `plotting_master.py`. Dit maakt het eenvoudig om nieuwe analyses toe te voegen.
        -   `sample_rate_dif/`: Bevat een klein, opzichzelfstaand script (`show_dif.py`) dat gebruikt werd om figuren te genereren die het effect van verschillende ADC-samplerates demonstreren.

## Hoe te gebruiken

### 1. Hardware & Firmware Installatie
-   **Hardware**: Je hebt twee complete ultrasone modules nodig, elk bestaande uit:
    -   TI MSP-EXP430F5529LP LaunchPad
    -   TI BOOSTXL-TUSS4440 BoosterPack
    -   Een 40 kHz ultrasone transducer
-   **Firmware**:
    1.  Open het `.ino`-bestand uit de `Energia/` map met de [Energia IDE](http://energia.nu/).
    2.  Installeer de meegeleverde (aangepaste) TUSS4440-librarybestanden in de `libraries`-map van Energia.
    3.  Flash de firmware naar beide MSP430-microcontrollers.

### 2. Data-acquisitie
1.  Verbind beide microcontrollers via USB met je PC. Noteer hun toegewezen COM-poorten.
2.  Open het `Python/meerdere_metingen.py` script.
3.  Pas de `MASTER_PORT` en `SLAVE_PORT` variabelen bovenaan het script aan met de juiste COM-poorten.
4.  Voer het script uit vanuit je terminal: `python meerdere_metingen.py`.
5.  Het script vraagt om een basisnaam voor de bestanden. Vervolgens voert het een serie geautomatiseerde metingen uit ("meta-runs") en slaat de data voor elke meta-run op in een uniek benoemd `.csv`-bestand.

### 3. Data-analyse en Visualisatie
1.  Navigeer naar de `Python/plotting/` directory.
2.  Open het `plotting_master.py` script. Dit is je hoofd-controlepaneel.
3.  **Selecteer Data**: Pas de `_CSV_BASE_FILES` lijst aan met de bestandsnamen van de `.csv`-data die je wilt analyseren.
4.  **Selecteer Plot-Modi**: Pas de `PLOT_MODES_TO_RUN` lijst aan om te specificeren welke analyses je wilt uitvoeren. Elke string in deze lijst komt overeen met een module in de `plot_modes/` map.
5.  **Configureer Analyse**: Het script maakt gebruik van `ANALYSIS_PROFILES` (bijv. voor 'aluminium' of 'g10') om verschillende parameters (zoals signaaldrempels) toe te passen op basis van het geanalyseerde materiaal. Je kunt deze profielen naar wens aanpassen.
6.  Voer het script uit vanuit je terminal: `python plotting_master.py`.
7.  Het script genereert en toont de geselecteerde plots.
