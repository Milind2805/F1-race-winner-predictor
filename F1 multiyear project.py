import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import logging
logging.disable(logging.CRITICAL)
import time
# Enable FastF1 cache
fastf1.Cache.enable_cache(r'd:\f1_cache')
races_2019=['Australian Grand Prix', 'Bahrain Grand Prix', 'Chinese Grand Prix', 'Azerbaijan Grand Prix', 'Spanish Grand Prix', 'Monaco Grand Prix', 'Canadian Grand Prix', 'French Grand Prix', 'Austrian Grand Prix', 'British Grand Prix', 'German Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 'Italian Grand Prix', 'Singapore Grand Prix', 'Russian Grand Prix', 'Japanese Grand Prix', 
            'Mexico City Grand Prix', 'United States Grand Prix', 'Brazilian Grand Prix', 'Abu Dhabi Grand Prix']
races_2020=['Austrian Grand Prix', 'Styrian Grand Prix', 'Hungarian Grand Prix', 'British Grand Prix', '70th Anniversary Grand Prix', 'Spanish Grand Prix', 'Belgian Grand Prix', 'Italian Grand Prix', 'Tuscan Grand Prix', 
            'Russian Grand Prix', 'Eifel Grand Prix', 'Portuguese Grand Prix', 'Emilia Romagna Grand Prix', 'Turkish Grand Prix', 'Bahrain Grand Prix', 'Sakhir Grand Prix', 'Abu Dhabi Grand Prix']
races_2021=['Bahrain Grand Prix', 'Emilia Romagna Grand Prix', 'Portuguese Grand Prix', 'Spanish Grand Prix', 'Monaco Grand Prix', 'Azerbaijan Grand Prix', 'French Grand Prix', 'Styrian Grand Prix', 'Austrian Grand Prix', 'British Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 'Italian Grand Prix',
            'Russian Grand Prix', 'Turkish Grand Prix', 'United States Grand Prix', 'Mexico City Grand Prix', 'Brazilian Grand Prix', 'Qatar Grand Prix', 'Saudi Arabian Grand Prix', 'Abu Dhabi Grand Prix']
races_2022=['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Emilia Romagna Grand Prix', 'Miami Grand Prix', 'Spanish Grand Prix', 'Monaco Grand Prix', 'Azerbaijan Grand Prix', 'Canadian Grand Prix', 'British Grand Prix', 'Austrian Grand Prix', 'French Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 'Italian Grand Prix', 'Singapore Grand Prix', 'Japanese Grand Prix',
            'United States Grand Prix', 'Mexico City Grand Prix', 'Brazilian Grand Prix', 'Abu Dhabi Grand Prix']
races_2023=['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Azerbaijan Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix', 'Spanish Grand Prix', 'Canadian Grand Prix', 'Austrian Grand Prix', 'British Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 'Italian Grand Prix', 'Singapore Grand Prix', 'Japanese Grand Prix',
            'Qatar Grand Prix', 'United States Grand Prix', 'Mexico City Grand Prix', 'Brazilian Grand Prix', 'Las Vegas Grand Prix', 'Abu Dhabi Grand Prix']
races_2024 = [
    'Bahrain Grand Prix', 'Saudi Arabian Grand Prix',
    'Australian Grand Prix', 'Japanese Grand Prix',
    'Chinese Grand Prix', 'Miami Grand Prix',
    'Emilia Romagna Grand Prix', 'Monaco Grand Prix',
    'Canadian Grand Prix', 'Spanish Grand Prix',
    'Austrian Grand Prix', 'British Grand Prix',
    'Hungarian Grand Prix', 'Belgian Grand Prix',
    'Dutch Grand Prix', 'Italian Grand Prix',
    'Azerbaijan Grand Prix', 'Singapore Grand Prix',
    'United States Grand Prix', 'Mexico City Grand Prix',
    'São Paulo Grand Prix', 'Las Vegas Grand Prix',
    'Qatar Grand Prix', 'Abu Dhabi Grand Prix'
]

races_2025 = [
    'Australian Grand Prix', 'Chinese Grand Prix',
    'Japanese Grand Prix', 'Bahrain Grand Prix',
    'Saudi Arabian Grand Prix', 'Miami Grand Prix',
    'Emilia Romagna Grand Prix', 'Monaco Grand Prix',
    'Spanish Grand Prix', 'Canadian Grand Prix',
    'Austrian Grand Prix', 'British Grand Prix',
    'Belgian Grand Prix', 'Hungarian Grand Prix',
    'Dutch Grand Prix', 'Italian Grand Prix',
    'Azerbaijan Grand Prix', 'Singapore Grand Prix',
    'United States Grand Prix', 'Mexico City Grand Prix',
    'São Paulo Grand Prix', 'Las Vegas Grand Prix',
    'Qatar Grand Prix', 'Abu Dhabi Grand Prix'
]
# races_2026_fallback = [
#     'Australian Grand Prix', 'Chinese Grand Prix', 'Japanese Grand Prix',
#     'Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix',
#     'Canadian Grand Prix', 'Monaco Grand Prix', 'Spanish Grand Prix',
#     'Austrian Grand Prix', 'British Grand Prix', 'Belgian Grand Prix',
#     'Hungarian Grand Prix', 'Dutch Grand Prix', 'Italian Grand Prix',
#     'Madrid Grand Prix', 'Azerbaijan Grand Prix', 'Singapore Grand Prix',
#     'United States Grand Prix', 'Mexico City Grand Prix',
#     'São Paulo Grand Prix', 'Las Vegas Grand Prix', 'Qatar Grand Prix',
#     'Abu Dhabi Grand Prix'
# ]
# try:
#     races_2026 = get_season_races(2026, fastf1)
#     if not races_2026:
#         raise ValueError("Empty schedule returned")
# except Exception as e:
#     print(f"⚠️ Could not fetch 2026 schedule live ({e}); using fallback list. "
#           f"Verify event names against fastf1.get_event_schedule(2026) before running.")
#     races_2026 = races_2026_fallback
all_seasons={2019: races_2019, 2020: races_2020, 2021: races_2021, 2022: races_2022, 2023: races_2023, 2024: races_2024, 2025: races_2025}
home_circuits={"HAM":"British Grand Prix","VER":"Dutch Grand Prix","LEC":"Monaco Grand Prix","VET":"German Grand Prix","SAI":"Spanish Grand Prix","ALO":"Spanish Grand Prix","STR":"Canadian Grand Prix","OCO":"French Grand Prix","NOR":"British Grand Prix","RUS":"British Grand Prix","PIA":"Australian Grand Prix",
               "TSU":"Japanese Grand Prix","GAS":"French Grand Prix","MAG":"Danish Grand Prix","ZHO":"Chinese Grand Prix","LAT":"Canadian Grand Prix","DEV":"Dutch Grand Prix","HUL":"German Grand Prix","SCH":"German Grand Prix","RIC":"Australian Grand Prix","GRO":"French Grand Prix","PER":"Mexico City Grand Prix","RAI":"Finnish Grand Prix", "ANT": "Italian Grand Prix","HAD": "French Grand Prix","COL": "Argentine Grand Prix","BOR": "São Paulo Grand Prix","LIN": "British Grand Prix","BEA": "British Grand Prix","LAW": "Australian Grand Prix"}
def get_weather(session):
    try:
        w = session.weather_data
        if w is None or w.empty:
            return {'Rainfall': 0, 'AirTemp': 20.0, 'TrackTemp': 30.0}
        return {
            'Rainfall': int(w['Rainfall'].any()),
            'AirTemp': round(w['AirTemp'].mean(), 1),
            'TrackTemp': round(w['TrackTemp'].mean(), 1)
        }
    except:
        return {'Rainfall': 0, 'AirTemp': 20.0, 'TrackTemp': 30.0}
def get_standings_before_race(season_results, race_name, year):
    """Returns driver points accumulated BEFORE the current race."""
    races_so_far = []
    for yr, races in all_seasons.items():
        if yr == year:
            races_so_far = races[:races.index(race_name)]
            break

    if not races_so_far:
        # First race of season — everyone starts at 0
        return {}

    past = season_results[season_results['Race'].isin(races_so_far)]
    standings = past.groupby('Abbreviation')['Points'].sum().to_dict()
    return standings
all_data = []
for year, races in all_seasons.items():
    print(f"\n{'='*40}")
    print(f"  Processing {year} season...")
    print(f"{'='*40}")
    
    cumulative_points = {}
    for race_name in races:
        try:
            # Load race session
            race_session = fastf1.get_session(year, race_name, 'R')
            race_session.load(telemetry=False, laps=False)

            results = race_session.results[[
                'Abbreviation', 'TeamName',
                'Position', 'GridPosition', 'Points'
            ]].copy()
            results['Race'] = race_name
            results['Year'] = year

            # Weather
            weather = get_weather(race_session)
            results['Rainfall'] = weather['Rainfall']
            results['AirTemp'] = weather['AirTemp']
            results['TrackTemp'] = weather['TrackTemp']

            # ── Standings BEFORE this race ──
            results['StandingPoints'] = results['Abbreviation'].map(
                lambda drv: cumulative_points.get(drv, 0)
            )

            # ── Home circuit ──
            results['HomeRace'] = results['Abbreviation'].apply(
                lambda drv: 1 if home_circuits.get(drv) == race_name else 0
            )

            # ── Quali session ──
            time.sleep(2)  # small delay before next API call
            quali_session = fastf1.get_session(year, race_name, 'Q')
            quali_session.load(telemetry=False, laps=False, weather=False)
            quali = quali_session.results[['Abbreviation', 'Position']].rename(
                columns={'Position': 'QualiPosition'}
            )
            results = pd.merge(results, quali, on='Abbreviation', how='left')
            for _, row in results.iterrows():
                drv = row['Abbreviation']
                pts = row['Points'] if not pd.isna(row['Points']) else 0
                cumulative_points[drv] = cumulative_points.get(drv, 0) + pts

            all_data.append(results)
            print(f"  ✅ {year} — {race_name}")

            time.sleep(3)  # delay between races to avoid rate limit

        except Exception as e:
            print(f"  ❌ {year} — {race_name}: {e}")
            time.sleep(10)  # longer delay after an error
# ── Save ──
final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv('f1_multiyear.csv', index=False)
print(f"\n✅ Done! Dataset shape: {final_df.shape}")
print(final_df.head())