# MS9 -- slope contrast: track clustering and season

Source: `outputs/figure_data/FigG_perdate_slopes_robust.csv`; RGT per overpass from kakhovka_atl13_pass_levels.parquet (companion repository).

- Headline difference of period medians: +3.223 cm/km (14 pre / 14 post overpasses).
- Cluster bootstrap by RGT (9 tracks, 10000 draws): +3.235 cm/km, 95 % [+2.171, +5.172].
- Tracks flown in both periods (5: [205, 609, 645, 989, 1209]): paired post − pre medians +2.58, +3.70, +5.13, +2.95, +5.58 cm/km; positive 5/5; median +3.700.
- Season: Apr–Oct +0.106 → +4.116 cm/km (n 8/10, post positive 10/10); Nov–Mar +0.090 → +2.399 cm/km (n 6/4, post positive 4/4).

Reading: the contrast is not an artefact of repeated ground tracks or of a seasonal imbalance between the two samples; it survives clustering by track, pairing within track, and stratification by season.
