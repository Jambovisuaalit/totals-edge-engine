import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import requests
import os
from nba_api.stats.endpoints import leaguegamefinder

# --- ASETUKSET (HAETAAN YMPÄRISTÖMUUTTUJISTA) ---
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHANNEL_ID = os.environ["TELEGRAM_CHAT_ID"]
ODDS_API_KEY = os.environ["ODDS_API_KEY"]

# --- DATA: HISTORIA ---
def fetch_history():
    print("🚀 Fetching history...")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=['2023-24', '2024-25'], league_id_nullable='00', season_type_nullable='Regular Season')
        games = gamefinder.get_data_frames()[0]
    except:
        return pd.DataFrame()

    games = games[games['SEASON_ID'].astype(str).str.startswith('2')].copy()
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games.sort_values(['TEAM_ID', 'GAME_DATE'])
    
    games['POSS'] = games['FGA'] + 0.44 * games['FTA'] - games['OREB'] + games['TOV']
    games['Pace'] = games['POSS']
    games['ORtg'] = (games['PTS'] / games['POSS']) * 100
    games['PTS_ALLOWED'] = games['PTS'] - games['PLUS_MINUS']
    games['DRtg'] = (games['PTS_ALLOWED'] / games['POSS']) * 100
    
    grp = games.groupby('TEAM_ID')
    games['rolling_ortg'] = grp['ORtg'].transform(lambda x: x.shift(1).rolling(10).mean())
    games['rolling_drtg'] = grp['DRtg'].transform(lambda x: x.shift(1).rolling(10).mean())
    games['rolling_pace'] = grp['Pace'].transform(lambda x: x.shift(1).rolling(10).mean())
    
    return games.dropna(subset=['rolling_ortg', 'rolling_drtg'])

# --- DATA: LIVE ODDS ---
def fetch_live_odds():
    print("💰 Fetching live odds...")
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?apiKey={ODDS_API_KEY}&regions=eu&markets=totals&oddsFormat=decimal"
    try:
        res = requests.get(url).json()
        games_list = []
        if isinstance(res, dict) and 'message' in res: return pd.DataFrame()
             
        for game in res:
            home = game['home_team']
            away = game['away_team']
            line = 0.0
            
            # KORJATTU LOOPPI TÄSSÄ
            for book in game['bookmakers']:
                for m in book['markets']:
                    if m['key'] == 'totals':
                        line = m['outcomes'][0]['point']
                        break
                if line > 0: break
            
            if line > 0: 
                games_list.append({'HomeTeam': home, 'AwayTeam': away, 'MarketLine': line})
        
        return pd.DataFrame(games_list)
    except:
        return pd.DataFrame()

# --- ENGINE ---
def run_engine():
    df = fetch_history()
    if df.empty: return
    
    matchups = pd.merge(df, df, on='GAME_ID', suffixes=('', '_OPP'))
    matchups = matchups[matchups['TEAM_ID'] != matchups['TEAM_ID_OPP']]
    
    features = ['rolling_ortg', 'rolling_drtg_OPP', 'rolling_pace', 'rolling_pace_OPP']
    model = xgb.XGBRegressor(n_estimators=500, max_depth=4).fit(matchups[features], matchups['PTS'])
    
    odds_df = fetch_live_odds()
    if odds_df.empty: return

    alerts = []
    for _, row in odds_df.iterrows():
        try:
            h_stats = df[df['TEAM_NAME'] == row['HomeTeam']].iloc[-1]
            a_stats = df[df['TEAM_NAME'] == row['AwayTeam']].iloc[-1]
            
            h_in = pd.DataFrame([{'rolling_ortg': h_stats['rolling_ortg'], 'rolling_drtg_OPP': a_stats['rolling_drtg'], 'rolling_pace': h_stats['rolling_pace'], 'rolling_pace_OPP': a_stats['rolling_pace']}])
            a_in = pd.DataFrame([{'rolling_ortg': a_stats['rolling_ortg'], 'rolling_drtg_OPP': h_stats['rolling_drtg'], 'rolling_pace': a_stats['rolling_pace'], 'rolling_pace_OPP': h_stats['rolling_pace']}])
            
            total = model.predict(h_in)[0] + model.predict(a_in)[0]
            edge = total - row['MarketLine']
            
            icon = "🟢 OVER" if edge > 4.0 else "🔴 UNDER" if edge < -4.0 else "⚪"
            if abs(edge) > 3.5:
                alerts.append(f"<b>{row['AwayTeam']} @ {row['HomeTeam']}</b>\nModel: {total:.1f} | Line: {row['MarketLine']}\nEdge: <b>{edge:+.1f}</b> {icon}")
        except: continue

    if alerts:
        msg = f"<b>[ TOTALS EDGE ENGINE ]</b>\nDate: {datetime.date.today()}\n\n" + "\n-------------------\n".join(alerts)
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHANNEL_ID, "text": msg, "parse_mode": "HTML"})

if __name__ == "__main__":
    run_engine()
