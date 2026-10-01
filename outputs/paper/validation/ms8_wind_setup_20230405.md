# MS8 -- wind setup over the liman at the 5 April 2023 Kherson anomaly

Source: outputs/tables/p0e_zone3_era5_hourly.csv (sha256 1a31018272cc2408) (ERA5 hourly via Open-Meteo, p0e; reanalysis, ~28 km, 10 cells 46.25-47.25 N, 31.5-32.25 E). Epochs from `outputs/figure_data/Fig17_colocated_perdate.csv`. Along-axis: 65 deg, +ve toward Kherson. Setup scale: F = 60 km, h = 5 m, C_D = 0.0015 (order of magnitude only).

## 2023-04-05 -- liman_mean

- observed SWOT - ICESat-2: -18.1 cm over dt = +14.2 h
- along-liman wind: -4.0 m/s at the ICESat-2 epoch -> +2.2 m/s at the SWOT epoch (change +6.3 m/s; max speed between 6.9 m/s)
- setup scale: -4 cm -> +1 cm (change +5 cm); inverse barometer +0.0 cm
- sign of the wind change does NOT match the observation

## 2023-04-05 -- cell_nearest_kherson

- observed SWOT - ICESat-2: -18.1 cm over dt = +14.2 h
- along-liman wind: -3.2 m/s at the ICESat-2 epoch -> +1.8 m/s at the SWOT epoch (change +5.0 m/s; max speed between 5.7 m/s)
- setup scale: -2 cm -> +1 cm (change +3 cm); inverse barometer -1.4 cm
- sign of the wind change does NOT match the observation

## 2023-05-04 -- liman_mean

- observed SWOT - ICESat-2: +4.4 cm over dt = +11.1 h
- along-liman wind: -3.1 m/s at the ICESat-2 epoch -> +0.2 m/s at the SWOT epoch (change +3.3 m/s; max speed between 3.7 m/s)
- setup scale: -2 cm -> +0 cm (change +2 cm); inverse barometer +3.6 cm
- sign of the wind change MATCHES the observation

## 2023-05-04 -- cell_nearest_kherson

- observed SWOT - ICESat-2: +4.4 cm over dt = +11.1 h
- along-liman wind: -2.5 m/s at the ICESat-2 epoch -> +0.6 m/s at the SWOT epoch (change +3.1 m/s; max speed between 4.0 m/s)
- setup scale: -1 cm -> +0 cm (change +1 cm); inverse barometer +3.3 cm
- sign of the wind change MATCHES the observation

## 2023-05-14 -- liman_mean

- observed SWOT - ICESat-2: +0.2 cm over dt = -2.3 h
- along-liman wind: +2.3 m/s at the ICESat-2 epoch -> +2.8 m/s at the SWOT epoch (change +0.6 m/s; max speed between 3.0 m/s)
- setup scale: +1 cm -> +2 cm (change +1 cm); inverse barometer -0.3 cm
- sign of the wind change MATCHES the observation

## 2023-05-14 -- cell_nearest_kherson

- observed SWOT - ICESat-2: +0.2 cm over dt = -2.3 h
- along-liman wind: +2.1 m/s at the ICESat-2 epoch -> +2.4 m/s at the SWOT epoch (change +0.3 m/s; max speed between 3.2 m/s)
- setup scale: +1 cm -> +1 cm (change +0 cm); inverse barometer -0.3 cm
- sign of the wind change MATCHES the observation

## Reading

Between the two epochs the along-liman wind turned from -4.0 m/s (blowing water away from Kherson, setdown) to +2.2 m/s (toward Kherson, setup). Wind setup therefore predicts a RISE of order +5 cm at Kherson across dt, the same sign as the gauge trend (+5.9 cm) and opposite to the observed -18.1 cm. Pressure changed by -0.0 hPa. On the ERA5 reanalysis, wind setup does not explain the anomaly; it makes the discrepancy larger. The anomaly remains unexplained. The manuscript sentence 'no wind or pressure record exists' should read: no station record; the ERA5 reanalysis over the liman (p0e) shows a wind change of the wrong sign.
