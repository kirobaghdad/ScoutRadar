from __future__ import annotations



try:
    import great_expectations as gx
except ModuleNotFoundError:  # pragma: no cover
    gx = None



REQUIRED_FIELDS = {
    "competitions": ["competition_id", "name", "type"],
    "clubs": ["club_id", "name", "domestic_competition_id"],
    "games": ["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"],
    "players": ["player_id", "name", "last_season", "current_club_id", "date_of_birth", "position"],
    "appearances": ["appearance_id", "game_id", "player_id", "date", "minutes_played"],
    "club_games": ["game_id", "club_id", "opponent_id", "hosting", "is_win"],
    "game_events": ["game_event_id", "game_id", "date", "minute", "type"],
    "game_lineups": ["game_lineups_id", "game_id", "player_id", "date", "club_id", "player_name"],
    "player_valuations": ["player_id", "date", "market_value_in_eur"],
    "transfers": ["player_id", "transfer_date", "from_club_id", "to_club_id"],
}

STRICT_UNIQUE_FIELDS = {
    "players": ["player_id", "url"],
    "clubs": ["club_id", "club_code", "url"],
    "competitions": ["competition_id", "url"],
    "games": ["game_id", "url"],
    "appearances": ["appearance_id"],
    "game_events": ["game_event_id"],
    "game_lineups": ["game_lineups_id"],
}

CANDIDATE_UNIQUE_FIELDS = {
    "players": {
        "name": "Player names can repeat across different people.",
        "player_code": "player_code behaves like a slug and may repeat for similar names.",
    },
    "clubs": {
        "name": "Club names are business-key candidates, not strict technical keys.",
    },
    "competitions": {
        "name": "Competition names may repeat across countries.",
        "competition_code": "Competition codes may repeat across countries.",
    },
}

SUBSET_UNIQUE_KEYS = {
    "player_valuations": ["player_id", "date"],
    "transfers": ["player_id", "transfer_date", "from_club_id", "to_club_id"],
    "club_games": ["game_id", "club_id"],
    "appearances": ["game_id", "player_id"],
    "game_lineups": ["game_id", "player_id", "club_id"],
    "games": ["competition_id", "season", "date", "home_club_id", "away_club_id"],
}

# Only use IQR on continuous business fields where extreme values are meaningful.
OUTLIER_FIELDS = {
    "players": ["age_at_last_season", "height_in_cm", "market_value_in_eur", "highest_market_value_in_eur"],
    "clubs": ["squad_size", "average_age", "national_team_players", "stadium_seats"],
    "games": ["attendance"],
    "player_valuations": ["market_value_in_eur"],
    "transfers": ["market_value_in_eur"],
}

TIME_COLUMNS = {
    "games": "date",
    "appearances": "date",
    "game_events": "date",
    "game_lineups": "date",
    "player_valuations": "date",
    "transfers": "transfer_date",
}

TIMELINESS_DUPLICATE_KEYS = {
    "games": ["competition_id", "season", "date", "home_club_id", "away_club_id"],
    "appearances": ["game_id", "player_id", "date"],
    "game_lineups": ["game_id", "player_id", "club_id", "date"],
    "player_valuations": ["player_id", "date"],
    "transfers": ["player_id", "transfer_date", "from_club_id", "to_club_id"],
}

SEASONAL_TIMELINESS_TABLES = {"games", "appearances", "game_events", "game_lineups"}
TIMELINESS_MIN_BASELINE_ROWS = 100
TIMELINESS_MONTHLY_DEVIATION_PCT = 40.0

DISTRIBUTION_NUMERIC_FIELDS = {
    "players": ["height_in_cm", "market_value_in_eur", "highest_market_value_in_eur"],
    "clubs": ["squad_size", "average_age", "foreigners_percentage", "national_team_players", "stadium_seats"],
    "games": ["home_club_goals", "away_club_goals", "home_club_position", "away_club_position", "attendance"],
    "appearances": ["minutes_played", "goals", "assists", "yellow_cards", "red_cards"],
    "club_games": ["own_goals", "opponent_goals", "own_position", "opponent_position"],
    "game_events": ["minute"],
    "player_valuations": ["market_value_in_eur"],
    "transfers": ["transfer_fee", "market_value_in_eur"],
}

DISTRIBUTION_CATEGORICAL_FIELDS = {
    "competitions": ["type", "sub_type", "confederation", "is_major_national_league"],
    "players": ["position", "sub_position", "foot"],
    "clubs": ["domestic_competition_id"],
    "games": ["competition_type", "home_club_formation", "away_club_formation"],
    "appearances": ["competition_id"],
    "club_games": ["hosting", "is_win"],
    "game_events": ["type"],
    "game_lineups": ["type", "position"],
    "player_valuations": ["player_club_domestic_competition_id"],
    "transfers": ["transfer_season"],
}

RELATIONSHIP_CORRELATION_PAIRS = {
    "players": [
        {
            "left": "market_value_in_eur",
            "right": "highest_market_value_in_eur",
            "description": "Current and peak market value should move together.",
        }
    ],
    "clubs": [
        {
            "left": "foreigners_number",
            "right": "foreigners_percentage",
            "description": "Foreign player count should strongly align with foreigner percentage.",
        },
        {
            "left": "squad_size",
            "right": "national_team_players",
            "description": "Larger squads often contain more national-team players.",
        },
    ],
    "games": [
        {
            "left": "home_club_goals",
            "right": "home_club_position",
            "description": "Better-ranked home teams tend to score more.",
        },
        {
            "left": "away_club_goals",
            "right": "away_club_position",
            "description": "Better-ranked away teams tend to score more.",
        },
        {
            "left": "attendance",
            "right": "home_club_position",
            "description": "Higher-ranked home teams often attract larger attendances.",
        },
    ],
    "appearances": [
        {
            "left": "minutes_played",
            "right": "goals",
            "description": "Longer playing time may weakly increase goal totals.",
        },
        {
            "left": "minutes_played",
            "right": "assists",
            "description": "Longer playing time may weakly increase assist totals.",
        },
        {
            "left": "minutes_played",
            "right": "yellow_cards",
            "description": "Longer playing time may slightly increase booking risk.",
        },
    ],
    "club_games": [
        {
            "left": "own_goals",
            "right": "is_win",
            "description": "Scoring more should strongly relate to winning.",
        },
        {
            "left": "opponent_goals",
            "right": "is_win",
            "description": "Conceding more should relate negatively to winning.",
        },
        {
            "left": "own_position",
            "right": "is_win",
            "description": "Better-ranked teams should generally win more often.",
        },
    ],
    "transfers": [
        {
            "left": "transfer_fee",
            "right": "market_value_in_eur",
            "description": "Transfer fee should positively relate to market value.",
        }
    ],
}

RELATIONSHIP_MIN_SHARED_ROWS = 100
RELATIONSHIP_STRONG_THRESHOLD = 0.6
RELATIONSHIP_FOREIGNERS_PCT_TOLERANCE = 1.0

API_FOOTBALL_FIXTURES_URL = "https://v3.football.api-sports.io/fixtures"
API_FOOTBALL_BIG_FIVE_LEAGUES = {
    "GB1": {"api_league_id": 39, "name": "Premier League", "country": "England"},
    "ES1": {"api_league_id": 140, "name": "La Liga", "country": "Spain"},
    "IT1": {"api_league_id": 135, "name": "Serie A", "country": "Italy"},
    "L1": {"api_league_id": 78, "name": "Bundesliga", "country": "Germany"},
    "FR1": {"api_league_id": 61, "name": "Ligue 1", "country": "France"},
}
DEFAULT_API_FOOTBALL_START_SEASON = 2018
DEFAULT_API_FOOTBALL_END_SEASON = 2022
DEFAULT_API_FOOTBALL_FIXTURE_PARAMS = {
    "league": 39,
    "season": DEFAULT_API_FOOTBALL_START_SEASON,
    "status": "FT-AET-PEN",
    "timezone": "Africa/Cairo",
}
API_REQUIRED_WRAPPER_FIELDS = ["get", "parameters", "errors", "results", "paging", "response"]
API_REQUIRED_FIXTURE_FIELDS = [
    "fixture_id",
    "fixture_date",
    "fixture_timestamp",
    "status_short",
    "status_long",
    "league_id",
    "league_name",
    "season",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "home_goals",
    "away_goals",
]
API_COMPLETED_STATUSES = {"FT", "AET", "PEN"}
DEFAULT_API_FOOTBALL_CACHE_DIR = "data/api_football"
DEFAULT_API_FOOTBALL_CACHE_PATH = f"{DEFAULT_API_FOOTBALL_CACHE_DIR}/api_football_fixtures_sample.json"
