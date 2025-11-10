#!/usr/bin/env bash
# Download cloud cover data for the example in cloud-cover.py
# using wget -n to avoid re-downloading if the file already exists.
wget -n 'https://archive-api.open-meteo.com/v1/archive?latitude=51.05&longitude=3.7167&start_date=2010-01-01&end_date=2020-01-01&hourly=cloud_cover&format=csv' -O cloud-cover-ghent-2010-2020.csv
