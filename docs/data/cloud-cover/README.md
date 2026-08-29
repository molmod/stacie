# Hourly Cloud Cover in Ghent

This dataset contains hourly cloud cover observations
(the percentage of the sky covered by clouds) in Ghent, Belgium,
from January 1, 2010 to January 1, 2020.
It is used in the STACIE documentation to demonstrate a correlation time analysis
of a time series that does not come from a molecular simulation.
The data are retrieved from the historical weather API of
[Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api).

## File Summary

- `cloud-cover-ghent-2010-2020.csv`: The hourly cloud cover percentages, as downloaded.
- `download.sh`: The script that retrieves the CSV file.

## Data Generation

The CSV file is downloaded with:

```bash
./download.sh
```

The script uses `wget -nc`, so it does not download the file again when it is already present.
