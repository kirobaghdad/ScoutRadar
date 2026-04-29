from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def create_synthetic_phase2_raw_dir(base_dir: str | Path, *, n_transfers: int = 24) -> Path:
    raw_dir = Path(base_dir) / "player_scores_data"
    raw_dir.mkdir(parents=True, exist_ok=True)

    competitions = [
        {
            "competition_id": "GB1",
            "competition_code": "premier-league",
            "name": "Premier League",
            "sub_type": "first_tier",
            "type": "domestic_league",
            "country_id": 1,
            "country_name": "England",
            "domestic_league_code": "GB1",
            "confederation": "europa",
            "url": "https://example.com/gb1",
            "is_major_national_league": True,
        },
        {
            "competition_id": "ES1",
            "competition_code": "laliga",
            "name": "LaLiga",
            "sub_type": "first_tier",
            "type": "domestic_league",
            "country_id": 2,
            "country_name": "Spain",
            "domestic_league_code": "ES1",
            "confederation": "europa",
            "url": "https://example.com/es1",
            "is_major_national_league": True,
        },
        {
            "competition_id": "IT1",
            "competition_code": "serie-a",
            "name": "Serie A",
            "sub_type": "first_tier",
            "type": "domestic_league",
            "country_id": 3,
            "country_name": "Italy",
            "domestic_league_code": "IT1",
            "confederation": "europa",
            "url": "https://example.com/it1",
            "is_major_national_league": True,
        },
        {
            "competition_id": "FR1",
            "competition_code": "ligue-1",
            "name": "Ligue 1",
            "sub_type": "first_tier",
            "type": "domestic_league",
            "country_id": 4,
            "country_name": "France",
            "domestic_league_code": "FR1",
            "confederation": "europa",
            "url": "https://example.com/fr1",
            "is_major_national_league": True,
        },
        {
            "competition_id": "L1",
            "competition_code": "bundesliga",
            "name": "Bundesliga",
            "sub_type": "first_tier",
            "type": "domestic_league",
            "country_id": 5,
            "country_name": "Germany",
            "domestic_league_code": "L1",
            "confederation": "europa",
            "url": "https://example.com/l1",
            "is_major_national_league": True,
        },
        {
            "competition_id": "PT1",
            "competition_code": "liga-portugal",
            "name": "Liga Portugal",
            "sub_type": "first_tier",
            "type": "domestic_league",
            "country_id": 6,
            "country_name": "Portugal",
            "domestic_league_code": "PT1",
            "confederation": "europa",
            "url": "https://example.com/pt1",
            "is_major_national_league": False,
        },
        {
            "competition_id": "NL1",
            "competition_code": "eredvisie",
            "name": "Eredivisie",
            "sub_type": "first_tier",
            "type": "domestic_league",
            "country_id": 7,
            "country_name": "Netherlands",
            "domestic_league_code": "NL1",
            "confederation": "europa",
            "url": "https://example.com/nl1",
            "is_major_national_league": False,
        },
    ]

    big_five_destinations = [
        (101, "Premier Town", "GB1"),
        (102, "Madrid Norte", "ES1"),
        (103, "Torino Verde", "IT1"),
        (104, "Paris Bleu", "FR1"),
        (105, "Rhein Stars", "L1"),
    ]
    source_clubs = [
        (201, "Porto B", "PT1"),
        (202, "Lisbon East", "PT1"),
        (203, "Rotterdam Waves", "NL1"),
    ]
    all_clubs = big_five_destinations + source_clubs + [(301, "Outside Club", "PT1")]

    clubs_rows = []
    for index, (club_id, name, competition_id) in enumerate(all_clubs):
        clubs_rows.append(
            {
                "club_id": club_id,
                "club_code": name.lower().replace(" ", "-"),
                "name": name,
                "domestic_competition_id": competition_id,
                "total_market_value": pd.NA,
                "squad_size": 23 + (index % 6),
                "average_age": 24.5 + (index % 4),
                "foreigners_number": 10 + (index % 5),
                "foreigners_percentage": 40.0 + (index * 2),
                "national_team_players": 3 + (index % 4),
                "stadium_name": f"{name} Stadium",
                "stadium_seats": 25000 + (index * 1500),
                "net_transfer_record": "€0",
                "coach_name": f"Coach {index}",
                "last_season": 2024,
                "filename": f"{name}.json",
                "url": f"https://example.com/clubs/{club_id}",
            }
        )

    players_rows = []
    transfers_rows = []
    valuations_rows = []
    appearances_rows = []
    games_rows = []
    club_games_rows = []

    game_id = 1
    player_positions = ["Attack", "Midfield", "Defender", "Goalkeeper"]
    player_sub_positions = ["Centre-Forward", "Central Midfield", "Centre-Back", "Goalkeeper"]

    for i in range(n_transfers):
        player_id = 1000 + i
        transfer_date = pd.Timestamp("2018-07-15") + pd.Timedelta(days=45 * i)
        success = i % 2 == 1
        destination_club_id, destination_name, destination_comp = big_five_destinations[i % len(big_five_destinations)]
        source_club_id, source_name, source_comp = source_clubs[i % len(source_clubs)]
        player_name = f"Player {i}"
        position = player_positions[i % len(player_positions)]
        sub_position = player_sub_positions[i % len(player_sub_positions)]
        base_value = 1_000_000 + (i * 25_000)

        players_rows.append(
            {
                "player_id": player_id,
                "first_name": "Player",
                "last_name": str(i),
                "name": player_name,
                "last_season": 2024,
                "current_club_id": destination_club_id,
                "player_code": f"player-{i}",
                "country_of_birth": "Syntheticland",
                "city_of_birth": "Test City",
                "country_of_citizenship": "Syntheticland",
                "date_of_birth": (pd.Timestamp("1995-01-01") + pd.Timedelta(days=30 * i)).strftime("%Y-%m-%d"),
                "sub_position": sub_position,
                "position": position,
                "foot": "right" if i % 2 == 0 else "left",
                "height_in_cm": 175 + (i % 15),
                "contract_expiration_date": pd.NA,
                "agent_name": "Test Agent",
                "image_url": "https://example.com/player.jpg",
                "url": f"https://example.com/player/{player_id}",
                "current_club_domestic_competition_id": destination_comp,
                "current_club_name": destination_name,
                "market_value_in_eur": base_value,
                "highest_market_value_in_eur": int(base_value * 1.5),
            }
        )

        transfers_rows.append(
            {
                "player_id": player_id,
                "transfer_date": transfer_date.strftime("%Y-%m-%d"),
                "transfer_season": f"{str(transfer_date.year % 100).zfill(2)}/{str((transfer_date.year + 1) % 100).zfill(2)}",
                "from_club_id": source_club_id,
                "to_club_id": destination_club_id,
                "from_club_name": source_name,
                "to_club_name": destination_name,
                "transfer_fee": int(base_value * (1.10 if success else 0.90)),
                "market_value_in_eur": base_value,
                "player_name": player_name,
            }
        )

        valuations_rows.extend(
            [
                {
                    "player_id": player_id,
                    "date": (transfer_date - pd.Timedelta(days=365)).strftime("%Y-%m-%d"),
                    "market_value_in_eur": int(base_value * 0.80),
                    "current_club_name": source_name,
                    "current_club_id": source_club_id,
                    "player_club_domestic_competition_id": source_comp,
                },
                {
                    "player_id": player_id,
                    "date": (transfer_date - pd.Timedelta(days=180)).strftime("%Y-%m-%d"),
                    "market_value_in_eur": int(base_value * 0.90),
                    "current_club_name": source_name,
                    "current_club_id": source_club_id,
                    "player_club_domestic_competition_id": source_comp,
                },
                {
                    "player_id": player_id,
                    "date": transfer_date.strftime("%Y-%m-%d"),
                    "market_value_in_eur": base_value,
                    "current_club_name": destination_name,
                    "current_club_id": destination_club_id,
                    "player_club_domestic_competition_id": destination_comp,
                },
                {
                    "player_id": player_id,
                    "date": (transfer_date + pd.Timedelta(days=365)).strftime("%Y-%m-%d"),
                    "market_value_in_eur": int(base_value * (1.20 if success else 0.85)),
                    "current_club_name": destination_name,
                    "current_club_id": destination_club_id,
                    "player_club_domestic_competition_id": destination_comp,
                },
                {
                    "player_id": player_id,
                    "date": (transfer_date + pd.Timedelta(days=700)).strftime("%Y-%m-%d"),
                    "market_value_in_eur": int(base_value * (1.30 if success else 0.75)),
                    "current_club_name": destination_name,
                    "current_club_id": destination_club_id,
                    "player_club_domestic_competition_id": destination_comp,
                },
            ]
        )

        if i == 0:
            valuations_rows.append(
                {
                    "player_id": player_id,
                    "date": (transfer_date + pd.Timedelta(days=850)).strftime("%Y-%m-%d"),
                    "market_value_in_eur": int(base_value * 1.35),
                    "current_club_name": destination_name,
                    "current_club_id": destination_club_id,
                    "player_club_domestic_competition_id": destination_comp,
                }
            )

        for match_index in range(8):
            appearance_date = transfer_date - pd.Timedelta(days=320 - (match_index * 30))
            appearances_rows.append(
                {
                    "appearance_id": f"{game_id}_{player_id}",
                    "game_id": game_id,
                    "player_id": player_id,
                    "player_club_id": source_club_id,
                    "player_current_club_id": source_club_id,
                    "date": appearance_date.strftime("%Y-%m-%d"),
                    "player_name": player_name,
                    "competition_id": source_comp,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "goals": 1 if position == "Attack" and match_index % 3 == 0 else 0,
                    "assists": 1 if position in {"Attack", "Midfield"} and match_index % 4 == 0 else 0,
                    "minutes_played": 90,
                }
            )
            game_id += 1

        post_matches = 22 if success else 8
        post_minutes = 90 if success else 60
        for match_index in range(post_matches):
            appearance_date = transfer_date + pd.Timedelta(days=25 + (match_index * 28))
            appearances_rows.append(
                {
                    "appearance_id": f"{game_id}_{player_id}",
                    "game_id": game_id,
                    "player_id": player_id,
                    "player_club_id": destination_club_id,
                    "player_current_club_id": destination_club_id,
                    "date": appearance_date.strftime("%Y-%m-%d"),
                    "player_name": player_name,
                    "competition_id": destination_comp,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "goals": 1 if success and position == "Attack" and match_index % 3 == 0 else 0,
                    "assists": 1 if success and position in {"Attack", "Midfield"} and match_index % 4 == 0 else 0,
                    "minutes_played": post_minutes,
                }
            )
            game_id += 1

        if i == 0:
            for extra_match in range(20):
                appearance_date = transfer_date + pd.Timedelta(days=760 + (extra_match * 10))
                appearances_rows.append(
                    {
                        "appearance_id": f"{game_id}_{player_id}",
                        "game_id": game_id,
                        "player_id": player_id,
                        "player_club_id": destination_club_id,
                        "player_current_club_id": destination_club_id,
                        "date": appearance_date.strftime("%Y-%m-%d"),
                        "player_name": player_name,
                        "competition_id": destination_comp,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "goals": 1,
                        "assists": 0,
                        "minutes_played": 90,
                    }
                )
                game_id += 1

    ineligible_player_id = 5000
    ineligible_transfer_date = pd.Timestamp("2021-01-15")
    players_rows.append(
        {
            "player_id": ineligible_player_id,
            "first_name": "Missing",
            "last_name": "Value",
            "name": "Missing Value",
            "last_season": 2024,
            "current_club_id": 101,
            "player_code": "missing-value",
            "country_of_birth": "Syntheticland",
            "city_of_birth": "Test City",
            "country_of_citizenship": "Syntheticland",
            "date_of_birth": "1996-06-01",
            "sub_position": "Centre-Forward",
            "position": "Attack",
            "foot": "right",
            "height_in_cm": 182,
            "contract_expiration_date": pd.NA,
            "agent_name": "Test Agent",
            "image_url": "https://example.com/player.jpg",
            "url": "https://example.com/player/5000",
            "current_club_domestic_competition_id": "GB1",
            "current_club_name": "Premier Town",
            "market_value_in_eur": 900000,
            "highest_market_value_in_eur": 1200000,
        }
    )
    transfers_rows.append(
        {
            "player_id": ineligible_player_id,
            "transfer_date": ineligible_transfer_date.strftime("%Y-%m-%d"),
            "transfer_season": "21/22",
            "from_club_id": 201,
            "to_club_id": 101,
            "from_club_name": "Porto B",
            "to_club_name": "Premier Town",
            "transfer_fee": 850000,
            "market_value_in_eur": 900000,
            "player_name": "Missing Value",
        }
    )
    valuations_rows.extend(
        [
            {
                "player_id": ineligible_player_id,
                "date": (ineligible_transfer_date - pd.Timedelta(days=180)).strftime("%Y-%m-%d"),
                "market_value_in_eur": 850000,
                "current_club_name": "Porto B",
                "current_club_id": 201,
                "player_club_domestic_competition_id": "PT1",
            },
            {
                "player_id": ineligible_player_id,
                "date": ineligible_transfer_date.strftime("%Y-%m-%d"),
                "market_value_in_eur": 900000,
                "current_club_name": "Premier Town",
                "current_club_id": 101,
                "player_club_domestic_competition_id": "GB1",
            },
        ]
    )

    transfers_rows.append(
        {
            "player_id": 6000,
            "transfer_date": "2023-01-10",
            "transfer_season": "22/23",
            "from_club_id": 201,
            "to_club_id": 101,
            "from_club_name": "Porto B",
            "to_club_name": "Premier Town",
            "transfer_fee": 1000000,
            "market_value_in_eur": 1000000,
            "player_name": "Future Transfer",
        }
    )
    transfers_rows.append(
        {
            "player_id": 6001,
            "transfer_date": "2021-02-10",
            "transfer_season": "20/21",
            "from_club_id": 202,
            "to_club_id": 301,
            "from_club_name": "Lisbon East",
            "to_club_name": "Outside Club",
            "transfer_fee": 1000000,
            "market_value_in_eur": 1000000,
            "player_name": "Outside League",
        }
    )
    transfers_rows.append(transfers_rows[0].copy())

    for club_id, club_name, competition_id in all_clubs:
        for month_index in range(72):
            match_date = pd.Timestamp("2017-01-15") + pd.DateOffset(months=month_index)
            own_goals = 2 if club_id in {101, 102, 103, 104, 105} else 1
            opponent_goals = 1 if month_index % 4 else 0
            is_win = int(own_goals > opponent_goals)

            games_rows.append(
                {
                    "game_id": game_id,
                    "competition_id": competition_id,
                    "season": match_date.year if match_date.month >= 7 else match_date.year - 1,
                    "round": f"Round {month_index + 1}",
                    "date": match_date.strftime("%Y-%m-%d"),
                    "home_club_id": club_id,
                    "away_club_id": 90000 + month_index,
                    "home_club_goals": own_goals,
                    "away_club_goals": opponent_goals,
                    "home_club_position": 3 + (month_index % 5),
                    "away_club_position": 10,
                    "home_club_manager_name": f"Coach {club_id}",
                    "away_club_manager_name": "Opponent Coach",
                    "stadium": f"{club_name} Arena",
                    "attendance": 20000 + (month_index * 50),
                    "referee": "Test Ref",
                    "url": f"https://example.com/game/{game_id}",
                    "home_club_formation": "4-3-3",
                    "away_club_formation": "4-4-2",
                    "home_club_name": club_name,
                    "away_club_name": "Opponent",
                    "aggregate": f"{own_goals}:{opponent_goals}",
                    "competition_type": "domestic_league",
                }
            )
            club_games_rows.append(
                {
                    "game_id": game_id,
                    "club_id": club_id,
                    "own_goals": own_goals,
                    "own_position": 3 + (month_index % 5),
                    "own_manager_name": f"Coach {club_id}",
                    "opponent_id": 90000 + month_index,
                    "opponent_goals": opponent_goals,
                    "opponent_position": 10,
                    "opponent_manager_name": "Opponent Coach",
                    "hosting": "Home",
                    "is_win": is_win,
                }
            )
            game_id += 1

    pd.DataFrame(competitions).to_csv(raw_dir / "competitions.csv", index=False)
    pd.DataFrame(clubs_rows).to_csv(raw_dir / "clubs.csv", index=False)
    pd.DataFrame(players_rows).to_csv(raw_dir / "players.csv", index=False)
    pd.DataFrame(transfers_rows).to_csv(raw_dir / "transfers.csv", index=False)
    pd.DataFrame(valuations_rows).to_csv(raw_dir / "player_valuations.csv", index=False)
    pd.DataFrame(appearances_rows).to_csv(raw_dir / "appearances.csv", index=False)
    pd.DataFrame(games_rows).to_csv(raw_dir / "games.csv", index=False)
    pd.DataFrame(club_games_rows).to_csv(raw_dir / "club_games.csv", index=False)

    api_fixture_rows = []
    fixture_id = 100000
    for season in range(2018, 2023):
        for club_id, club_name, competition_id in all_clubs:
            fixture_id += 1
            api_fixture_rows.append(
                {
                    "fixture": {
                        "id": fixture_id,
                        "date": f"{season}-09-01T15:00:00+00:00",
                        "timestamp": int(pd.Timestamp(f"{season}-09-01").timestamp()),
                        "status": {"short": "FT", "long": "Match Finished", "elapsed": 90},
                    },
                    "league": {"id": 1, "name": competition_id, "country": "Syntheticland", "season": season},
                    "teams": {
                        "home": {"id": club_id, "name": club_name, "winner": True},
                        "away": {"id": 900000 + fixture_id, "name": f"API Opponent {fixture_id}", "winner": False},
                    },
                    "goals": {"home": 2, "away": 1},
                }
            )
    api_cache_dir = Path(base_dir) / "api_football"
    api_cache_dir.mkdir(parents=True, exist_ok=True)
    (api_cache_dir / "api_football_fixtures_synthetic.json").write_text(
        json.dumps({"response": api_fixture_rows}),
        encoding="utf-8",
    )

    return raw_dir
