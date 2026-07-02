# Profielentool

Startproject voor een Streamlit-app om dwarsprofielen te maken op basis van CSV- en GIS-bestanden.

## Wat staat er nu klaar?

- Een werkende Streamlit starter-app in `app.py`
- Basis projectstructuur met `src`, `data`, `tests` en `docs`
- GIS/CSV dependencies in `requirements.txt`
- Een voorbeeld CSV in `data/input/sample_points.csv`

## Projectstructuur

```text
profielentool/
  app.py
  requirements.txt
  README.md
  src/profielentool/
  data/input/
  data/output/
  tests/
  docs/
```

## Eerste keer opzetten (Windows PowerShell)

1. Maak een virtuele omgeving:

   ```powershell
   py -3 -m venv .venv
   ```

2. Activeer de omgeving:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Installeer dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Start de app:

   ```powershell
   streamlit run app.py
   ```

5. Open de URL die in de terminal verschijnt (meestal `http://localhost:8501`).

## Opmerking over GIS op Windows

Sommige GIS libraries (zoals Fiona en Rasterio) kunnen op bepaalde machines extra systeemafhankelijkheden nodig hebben. Als installatieproblemen optreden, lossen we die samen stap voor stap op.

## Volgende stap

Geef door welke inputbestanden je als eerste wilt ondersteunen (bijvoorbeeld CSV + shapefile met assen), dan bouwen we de eerste profielberekening.
