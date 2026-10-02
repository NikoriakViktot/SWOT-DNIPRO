# MS6C — provenance of the v5 S3.3 crossing numbers

Baseline guard (cached pairs regrouped under the ms6b rule == ms6b tables, row by row): legacy_ZONE_1to4: 327 vs 327 rows, identical=True; paper_RFDE: 316 vs 316 rows, identical=True.
5 June 2023 S2 pool at 100 m: 2,395 km² (v5: 2 299 km²).
Variants evaluated: 1620; lake variants: 72.

## G1 (32 / 70 / 241 ⇒ 343, RiverSP): 0 variant(s) pass

No variant passes G1; the ten closest:

| variant                                                            |   n1 |   n13 |   n310 |   pre |   trans_0901 |   post_0901 |   trans_0908 |   post_0908 |   foot_SA2 |   foot_Rcore |   foot_Z1 |   foot_pool_20230605 |   all_med |   all_nmad | G2_post_cuts   | G3_footprints   |
|:-------------------------------------------------------------------|-----:|------:|-------:|------:|-------------:|------------:|-------------:|------------:|-----------:|-------------:|----------:|---------------------:|----------:|-----------:|:---------------|:----------------|
| all|q1_nodark|r200|Z1234|union_at_crossing|ovp_pass|h24            |   33 |    68 |    243 |     8 |            8 |          17 |            8 |          17 |          3 |            1 |        17 |                    2 |       1.7 |        9.3 |                |                 |
| all|q1_nodark|r200|Z1234|union_at_crossing|ovp_date|h24            |   33 |    68 |    243 |     8 |            8 |          17 |            8 |          17 |          3 |            1 |        17 |                    2 |       1.7 |        9.3 |                |                 |
| all|q1_nodark|r200|Z1234|union_at_node|ovp_pass|h24                |   33 |    68 |    243 |     8 |            8 |          17 |            8 |          17 |          3 |            1 |        17 |                    2 |       1.7 |        9.3 |                |                 |
| all|q1_nodark|r200|Z1234|union_at_node|ovp_date|h24                |   33 |    68 |    243 |     8 |            8 |          17 |            8 |          17 |          3 |            1 |        17 |                    2 |       1.7 |        9.3 |                |                 |
| all|q1_nodark|r200|Z1234|perzone_dedup_ovp_pass|ovp_pass|h24       |   33 |    68 |    243 |     8 |            8 |          17 |            8 |          17 |          3 |            1 |        17 |                    2 |       1.7 |        9.3 |                |                 |
| all|q1_nodark|r200|Z1234|perzone_dedup_ovp_pass|ovp_pass_reach|h24 |   33 |    68 |    243 |     8 |            8 |          17 |            8 |          17 |          3 |            2 |        17 |                    3 |       1.2 |        8.4 |                |                 |
| all|q1_nodark|r200|Z123_master|union_at_crossing|ovp_pass|h24      |   33 |    68 |    239 |     8 |            8 |          17 |            8 |          17 |          3 |            1 |        17 |                    2 |       1.7 |        9.3 |                |                 |
| all|q1_nodark|r200|Z123_master|union_at_crossing|ovp_date|h24      |   33 |    68 |    239 |     8 |            8 |          17 |            8 |          17 |          3 |            1 |        17 |                    2 |       1.7 |        9.3 |                |                 |
| all|q1_nodark|r200|Z123_master|union_at_node|ovp_pass|h24          |   33 |    68 |    243 |     8 |            8 |          17 |            8 |          17 |          3 |            1 |        17 |                    2 |       1.7 |        9.3 |                |                 |
| all|q1_nodark|r200|Z123_master|union_at_node|ovp_date|h24          |   33 |    68 |    243 |     8 |            8 |          17 |            8 |          17 |          3 |            1 |        17 |                    2 |       1.7 |        9.3 |                |                 |

## G2 (9 / 6 / 17): 0 variant(s) reproduce it on their own; 0 of them also pass G1

## G3 (footprint 21): 12 variant(s) reproduce it on their own; 0 of them also pass G1

| variant                                                     |   n1 |   n13 |   n310 | G3_footprints   |
|:------------------------------------------------------------|-----:|------:|-------:|:----------------|
| all|q2_dark|r500|Z123_master|perzone_concat|ovp_pass|h24    |   39 |    84 |    282 | Z1              |
| all|q2_dark|r500|Z123_master|perzone_concat|ovp_date|h24    |   39 |    84 |    282 | Z1              |
| latest|q2_dark|r500|Z123_master|perzone_concat|ovp_pass|h24 |   39 |    84 |    282 | Z1              |
| latest|q2_dark|r500|Z123_master|perzone_concat|ovp_date|h24 |   39 |    84 |    282 | Z1              |
| all|q2_dark|r200|Z14|perzone_dedup_exact|ovp_pass|h24       |   22 |    54 |    175 | Z1              |
| all|q2_dark|r200|Z14|perzone_dedup_exact|ovp_date|h24       |   22 |    54 |    175 | Z1              |
| latest|q2_dark|r200|Z14|perzone_dedup_exact|ovp_pass|h24    |   22 |    54 |    175 | Z1              |
| latest|q2_dark|r200|Z14|perzone_dedup_exact|ovp_date|h24    |   22 |    54 |    175 | Z1              |
| all|q2_dark|r200|Z1234|perzone_concat|ovp_pass|sameday      |   31 |   172 |    386 | Z1              |
| all|q2_dark|r200|Z1234|perzone_concat|ovp_date|sameday      |   31 |   172 |    386 | Z1              |
| latest|q2_dark|r200|Z1234|perzone_concat|ovp_pass|sameday   |   31 |   172 |    386 | Z1              |
| latest|q2_dark|r200|Z1234|perzone_concat|ovp_date|sameday   |   31 |   172 |    386 | Z1              |

## G4 (8 small-lake pairs): 0 lake variant(s)

| universe   | versions   | window   | quality   | construction   |   n_small |   n_large |   small_med |   small_nmad |   large_med |   large_nmad |
|:-----------|:-----------|:---------|:----------|:---------------|----------:|----------:|------------:|-------------:|------------:|-------------:|
| Z14        | all        | h24      | all       | segments       |         7 |        25 |         8   |          7.5 |       118.7 |        201.8 |
| Z14        | all        | h24      | all       | cells500       |         7 |        25 |         4   |         11.8 |       113.9 |        161.7 |
| Z14        | all        | h24      | q<=1      | segments       |         7 |        25 |         8   |          7.5 |       118.7 |        201.8 |
| Z14        | all        | h24      | q<=1      | cells500       |         7 |        25 |         4   |         11.8 |       113.9 |        161.7 |
| Z14        | latest     | h24      | all       | segments       |         6 |        20 |         6.3 |         17.6 |       101.2 |        183.7 |
| Z14        | latest     | h24      | all       | cells500       |         6 |        20 |         4.2 |         16.8 |        79.5 |        138.8 |
| Z14        | latest     | h24      | q<=1      | segments       |         6 |        20 |         6.3 |         17.6 |       101.2 |        183.7 |
| Z14        | latest     | h24      | q<=1      | cells500       |         6 |        20 |         4.2 |         16.8 |        79.5 |        138.8 |

