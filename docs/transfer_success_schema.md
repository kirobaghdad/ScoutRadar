# Transfer Success Modeling Schema

This schema defines a transfer-level dataset where each row is one transfer event from `transfers_clean.csv`.

## Row Grain

- One row = one player transfer (`player_id`, `transfer_date`, `from_club_id`, `to_club_id`).

## Target

- `transfer_success` (`int8`)
- Allowed values:
  - `1` = Success
  - `0` = Failure

## Target Rule (2-3 season window after transfer)

Use this deterministic rule:

- `criteria_minutes` = 1 if average minutes played per post-transfer season >= `900`, else 0.
- `criteria_value` = 1 if market value at end of window >= market value at transfer date, else 0.
- `criteria_position_kpi` = 1 if position KPI threshold is met, else 0.

Then:

- `transfer_success = 1` if at least 2 of 3 criteria are true.
- `transfer_success = 0` otherwise.

## Position KPI Rule

- Attack (`position == "Attack"`): post-transfer average `goals_per90 >= 0.35` OR `assists_per90 >= 0.20`
- Midfield (`position == "Midfield"`): post-transfer average `assists_per90 >= 0.18`
- Defender (`position == "Defender"`): post-transfer average `goals_per90 >= 0.08`
- Goalkeeper (`position == "Goalkeeper"`): post-transfer average `minutes_per_season >= 1800`

## Feature Schema

| Feature                        | Dtype            | Source                                                           | Definition                                                 |
| ------------------------------ | ---------------- | ---------------------------------------------------------------- | ---------------------------------------------------------- |
| `player_id`                    | `int64`          | `transfers_clean.player_id`                                      | Player identifier                                          |
| `transfer_date`                | `datetime64[ns]` | `transfers_clean.transfer_date`                                  | Transfer date                                              |
| `transfer_season`              | `int16`          | `transfers_clean.transfer_season`                                | Transfer season                                            |
| `from_club_id`                 | `int32`          | `transfers_clean.from_club_id`                                   | Selling club                                               |
| `to_club_id`                   | `int32`          | `transfers_clean.to_club_id`                                     | Buying club                                                |
| `transfer_fee_eur`             | `float64`        | `transfers_clean.transfer_fee`                                   | Parsed to numeric EUR                                      |
| `market_value_at_transfer_eur` | `float64`        | `transfers_clean.market_value_in_eur` or nearest from valuations | Market value at transfer time                              |
| `age_at_transfer`              | `float32`        | `players_clean.date_of_birth` + `transfer_date`                  | Age in years at transfer                                   |
| `primary_position`             | `category`       | `players_clean.position`                                         | Main position                                              |
| `secondary_position`           | `category`       | `players_clean.sub_position`                                     | Secondary position                                         |
| `nationality`                  | `category`       | `players_clean.country_of_citizenship`                           | Citizenship                                                |
| `height_cm`                    | `float32`        | `players_clean.height_in_cm`                                     | Player height                                              |
| `foot`                         | `category`       | `players_clean.foot`                                             | Preferred foot                                             |
| `pre_minutes_total`            | `float32`        | `appearances_clean`                                              | Minutes in pre-transfer window                             |
| `pre_goals_total`              | `float32`        | `appearances_clean`                                              | Goals in pre-transfer window                               |
| `pre_assists_total`            | `float32`        | `appearances_clean`                                              | Assists in pre-transfer window                             |
| `pre_yellow_cards_total`       | `float32`        | `appearances_clean`                                              | Yellow cards in pre-transfer window                        |
| `pre_red_cards_total`          | `float32`        | `appearances_clean`                                              | Red cards in pre-transfer window                           |
| `pre_goals_per90`              | `float32`        | engineered                                                       | `90 * pre_goals_total / pre_minutes_total`                 |
| `pre_assists_per90`            | `float32`        | engineered                                                       | `90 * pre_assists_total / pre_minutes_total`               |
| `pre_goal_contrib_per90`       | `float32`        | engineered                                                       | `pre_goals_per90 + pre_assists_per90`                      |
| `from_league_code`             | `category`       | `clubs_clean.domestic_competition_id` via `from_club_id`         | Seller league code                                         |
| `to_league_code`               | `category`       | `clubs_clean.domestic_competition_id` via `to_club_id`           | Buyer league code                                          |
| `from_is_top5`                 | `int8`           | engineered                                                       | 1 if `from_league_code` in {`GB1`,`ES1`,`IT1`,`FR1`,`L1`}  |
| `to_is_top5`                   | `int8`           | engineered                                                       | 1 if `to_league_code` in {`GB1`,`ES1`,`IT1`,`FR1`,`L1`}    |
| `league_tier_delta`            | `int8`           | engineered                                                       | Mapping-based strength delta (`to` minus `from`)           |
| `from_club_avg_age`            | `float32`        | `clubs_clean.average_age`                                        | Selling club avg age                                       |
| `to_club_avg_age`              | `float32`        | `clubs_clean.average_age`                                        | Buying club avg age                                        |
| `from_club_squad_size`         | `float32`        | `clubs_clean.squad_size`                                         | Selling club squad size                                    |
| `to_club_squad_size`           | `float32`        | `clubs_clean.squad_size`                                         | Buying club squad size                                     |
| `to_club_prev_season_rank`     | `float32`        | API merged standings                                             | Destination previous season rank                           |
| `to_club_current_season_ppg`   | `float32`        | API merged fixtures                                              | Destination current season points per game before transfer |
| `has_api_team_match`           | `int8`           | merge metadata                                                   | 1 if destination club matched API team                     |
| `has_api_prev_standing`        | `int8`           | merge metadata                                                   | 1 if destination previous-season standing exists           |
| `has_api_current_form`         | `int8`           | merge metadata                                                   | 1 if destination current-season metrics exist              |

## Post-Transfer Columns (for Target Construction Only)

These columns are computed during label generation and should not be used as model inputs:

| Column                        | Dtype     | Definition                                         |
| ----------------------------- | --------- | -------------------------------------------------- |
| `post_minutes_avg_per_season` | `float32` | Avg post-transfer minutes per season (2-3 seasons) |
| `post_goals_per90_avg`        | `float32` | Avg post-transfer goals/90                         |
| `post_assists_per90_avg`      | `float32` | Avg post-transfer assists/90                       |
| `market_value_end_window_eur` | `float64` | Market value at end of 2-3 season window           |
| `criteria_minutes`            | `int8`    | Minutes criterion outcome                          |
| `criteria_value`              | `int8`    | Market value criterion outcome                     |
| `criteria_position_kpi`       | `int8`    | Position KPI criterion outcome                     |
| `transfer_success`            | `int8`    | Final binary target                                |

## Recommended Train Input Set

Use all feature columns except identifiers and leakage-prone columns:

- Drop: `player_id`, `transfer_date`, `from_club_id`, `to_club_id`
- Keep: all other feature columns in the Feature Schema table
- Target: `transfer_success`

## Class Distribution Check

After creating labels:

- Report counts and ratio of `transfer_success` values.
- If minority class < 35%, apply class balancing:
  - `class_weight='balanced'` for Logistic Regression.
  - `class_weight='balanced_subsample'` for Random Forest.
