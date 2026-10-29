# Data sources

All data in this project is downloaded by `src/download.py`. Raw files are saved in
`data/raw/` with the download date in the file name and are never edited. Every
download is also recorded in `data/raw/download_log_<date>.json` (URLs, record
counts, the World Bank's own "last updated" date, and any errors).

The snapshot committed to this repository was downloaded on **2026-10-29**.
Running the pipeline again on a later date downloads fresh files. The World Bank
revises its data, so a later download can give slightly different numbers.

## 1. Country data: World Bank World Development Indicators (API)

| Variable in this project | Indicator code | Official name | Original source (from World Bank metadata) |
|---|---|---|---|
| `inflation` | `FP.CPI.TOTL.ZG` | Inflation, consumer prices (annual %) | International Financial Statistics database, International Monetary Fund (IMF) |
| `gdp_growth` | `NY.GDP.MKTP.KD.ZG` | GDP growth (annual %) | Country official statistics, national statistical organizations and/or central banks; OECD national accounts data files; World Bank staff estimates |
| `fx_rate` (used to compute `fx_change`) | `PA.NUS.FCRF` | Official exchange rate (LCU per US$, period average) | International Financial Statistics database, International Monetary Fund (IMF) |

- **API URL pattern:** `https://api.worldbank.org/v2/country/{countries}/indicator/{code}?format=json&date=2000:2026&per_page=1000&page={n}`
  (all pages are followed until the last page).
- **Countries (28, ISO3):** UZB, KAZ, KGZ, TJK, GEO, ARM, TUR, POL, ROU, EGY, MAR, NGA, KEN, GHA, ZAF, IND, PAK, BGD, IDN, PHL, VNM, THA, MYS, BRA, MEX, COL, PER, CHL.
- **Years requested:** 2000 to 2026. The API returned years 2000 to 2025 (728 records per indicator = 28 countries × 26 years, including empty values).
- **World Bank "last updated" date:** 2026-10-08 (all three indicators).
- **Download date:** 2026-10-29.
- **Raw files:** `data/raw/wb_<code>_2026-10-29.json` (data) and `data/raw/wb_<code>_metadata_2026-10-29.json` (indicator definition and source).
- **Empty values in the download:** inflation is empty for 12 country-years (Uzbekistan 11, Tajikistan 1); the exchange rate is empty for 13 (Uzbekistan 12, Vietnam 1); GDP growth has none.
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0), as stated in the World Bank's terms of use for datasets.
- **How to cite:** World Bank (2026). *World Development Indicators*: FP.CPI.TOTL.ZG, NY.GDP.MKTP.KD.ZG, PA.NUS.FCRF. Washington, DC: The World Bank. Retrieved 29 October 2026 from https://api.worldbank.org/v2/. License: CC BY 4.0.

## 2. Global commodity prices: World Bank "Pink Sheet"

- **Source used:** World Bank Commodity Price Data (The Pink Sheet), *CMO Historical Data Monthly* (Excel).
- **How the link was found:** `src/download.py` reads https://www.worldbank.org/en/research/commodity-markets and takes the link to `CMO-Historical-Data-Monthly.xlsx`.
- **File URL used:** https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx
- **Sheet used:** `Monthly Indices` (monthly indices in nominal US dollars, 2010 = 100). The file says "Updated on October 02, 2026".
- **Series used:** Energy, Food, Agriculture and Fertilizers (all four are in the file).
- **Months covered:** 1960M01 to 2026M09. 2026 is incomplete, so only full calendar years are turned into annual averages (see METHODS.md).
- **Download date:** 2026-10-29.
- **Raw file:** `data/raw/CMO-Historical-Data-Monthly_2026-10-29.xlsx`.
- **Note:** the file also contains a sheet called `Mismatch Details`, a World Bank comparison between this release and an earlier one (mostly blank Barley and Sorghum prices after 2020, plus 10 changed individual prices). It concerns individual prices in the `Monthly Prices` sheet, not the indices this project uses, and is not used.
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0), as stated in the World Bank's terms of use for datasets.
- **How to cite:** World Bank (2026). *World Bank Commodity Price Data (The Pink Sheet)*, CMO Historical Data Monthly, updated 2 October 2026. Washington, DC: The World Bank. Retrieved 29 October 2026 from https://www.worldbank.org/en/research/commodity-markets. License: CC BY 4.0.

## 3. Fallback (not used)

If the Pink Sheet download fails, `src/download.py` falls back to FRED CSV files (IMF
Primary Commodity Prices, no API key needed):

- Food: https://fred.stlouisfed.org/graph/fredgraph.csv?id=PFOODINDEXM
- Energy: https://fred.stlouisfed.org/graph/fredgraph.csv?id=PNRGINDEXM

**The fallback was not needed for the 2026-10-29 download.** The download log records
which source was used each time (`"commodities": {"source": ...}`).
If the fallback is ever used, cite it as: International Monetary Fund, *Global Price of
Food Index* (PFOODINDEXM) and *Global Price of Energy Index* (PNRGINDEXM), retrieved
from FRED, Federal Reserve Bank of St. Louis.

## Download history

- First attempt (2026-10-29, earlier session): every host was blocked by the
  computing environment's network settings (HTTP 403 from the proxy). No data was
  downloaded and nothing was substituted.
- Second attempt (2026-10-29, after network access was allowed): every download
  succeeded. This is the data used in all results.
