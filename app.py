# -*- coding: utf-8 -*-
# =============================================
# Gradio App for Chess Game Analysis - Lichess API Version
# v18: Manual progress updates instead of track_tqdm to fix IndexError.
# =============================================

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta, timezone
import time
import re
import traceback

# --- Configuration ---
# No Streamlit config needed

# --- Constants & Defaults ---
TIME_PERIOD_OPTIONS = { "Last Month": timedelta(days=30), "Last 3 Months": timedelta(days=90), "Last Year": timedelta(days=365), "Last 3 Years": timedelta(days=3*365) }
DEFAULT_TIME_PERIOD = "Last Year"
PERF_TYPE_OPTIONS_SINGLE = ['Bullet', 'Blitz', 'Rapid']
DEFAULT_PERF_TYPE = 'Bullet'
DEFAULT_RATED_ONLY = True
ECO_CSV_PATH = "eco_to_opening.csv"
TITLES_TO_ANALYZE = ['GM', 'IM', 'FM', 'CM', 'WGM', 'WIM', 'WFM', 'WCM', 'NM']

# =============================================
# Helper Function: Categorize Time Control (Correct)
# =============================================
# ... (Function identical to v17 - Assumed correct now) ...
def categorize_time_control(tc_str, speed_info):
    if isinstance(speed_info, str) and speed_info in ['bullet', 'blitz', 'rapid', 'classical', 'correspondence']: return speed_info.capitalize()
    if not isinstance(tc_str, str) or tc_str in ['-', '?', 'Unknown','Correspondence']: return 'Unknown' if tc_str!='Correspondence' else 'Correspondence'
    if '+' in tc_str:
        try: parts=tc_str.split('+');
             if len(parts)==2: base=int(parts[0]); increment=int(parts[1]); total=base+40*increment
             else: return 'Unknown'
        except(ValueError,IndexError): return 'Unknown'
        if total>=1500: return 'Classical';
        if total>=480: return 'Rapid';
        if total>=180: return 'Blitz';
        if total>0 : return 'Bullet';
        return 'Unknown'
    else:
        try: base=int(tc_str)
             if base>=1500: return 'Classical';
             if base>=480: return 'Rapid';
             if base>=180: return 'Blitz';
             if base>0 : return 'Bullet';
             return 'Unknown'
        except ValueError: tc_lower=tc_str.lower();
             if 'classical' in tc_lower: return 'Classical';
             if 'rapid' in tc_lower: return 'Rapid';
             if 'blitz' in tc_lower: return 'Blitz';
             if 'bullet' in tc_lower: return 'Bullet';
             return 'Unknown'

# =============================================
# Helper Function: Load ECO Mapping (Unchanged)
# =============================================
ECO_MAPPING = {}
try:
    df_eco_global = pd.read_csv(ECO_CSV_PATH)
    if "ECO Code" in df_eco_global.columns and "Opening Name" in df_eco_global.columns:
        ECO_MAPPING = df_eco_global.drop_duplicates(subset=['ECO Code']).set_index('ECO Code')['Opening Name'].to_dict()
        print(f"OK: Loaded {len(ECO_MAPPING)} ECO mappings.")
    else: print(f"WARN: ECO file '{ECO_CSV_PATH}' missing columns.")
except FileNotFoundError: print(f"WARN: ECO file '{ECO_CSV_PATH}' not found.")
except Exception as e: print(f"WARN: Error loading ECO file: {e}")

# =============================================
# API Data Loading and Processing Function (Manual Progress Update)
# =============================================
# Removed @gr.Progress decorator here, will pass progress object manually
def load_from_lichess_api(username: str, time_period_key: str, perf_type: str, rated: bool, eco_map: dict, progress=None): # Changed progress to optional default None
    """ Fetches and processes Lichess games with MANUAL progress updates. """
    if not username: return pd.DataFrame(), "⚠️ Enter username."
    if not perf_type: return pd.DataFrame(), "⚠️ Select game type."
    if progress: progress(0, desc="Initializing...") # Check if progress object exists
    username_lower=username.lower(); status_message=f"Fetching {perf_type} games..."
    if progress: progress(0.1, desc=status_message);
    since_timestamp_ms=None; time_delta=TIME_PERIOD_OPTIONS.get(time_period_key)
    if time_delta: start_date=datetime.now(timezone.utc)-time_delta; since_timestamp_ms=int(start_date.timestamp()*1000)
    api_params={"rated":str(rated).lower(), "perfType":perf_type.lower(), "opening":"true", "moves":"false", "tags":"false", "pgnInJson":"false" }
    if since_timestamp_ms: api_params["since"]=since_timestamp_ms
    api_url=f"https://lichess.org/api/games/user/{username}"; headers={"Accept":"application/x-ndjson"}
    all_games_data=[]; error_counter=0; lines_processed=0
    try:
        response=requests.get(api_url, params=api_params, headers=headers, stream=True); response.raise_for_status()
        if progress: progress(0.3, desc="Processing stream...")
        # Estimate total iterations for progress (difficult for streams, use a large number or steps)
        # Let's update every N lines instead.
        update_interval = 50 # Update progress every 50 games processed

        for line in response.iter_lines():
            if line:
                lines_processed += 1; game_data_raw=line.decode('utf-8'); game_data=None;
                # --- Manual Progress Update ---
                if progress and lines_processed % update_interval == 0:
                    # Simple pulsing progress indication
                    progress(0.3 + (lines_processed % (update_interval * 10)) / (update_interval * 20.0),
                             desc=f"Processing game ~{lines_processed}...")
                # --- End Manual Progress ---
                try:
                    game_data=json.loads(game_data_raw); white_info=game_data.get('players',{}).get('white',{}); black_info=game_data.get('players',{}).get('black',{})
                    white_user=white_info.get('user',{}); black_user=black_info.get('user',{}); opening_info=game_data.get('opening',{}); clock_info=game_data.get('clock')
                    game_id=game_data.get('id','N/A'); created_at_ms=game_data.get('createdAt'); game_date=pd.to_datetime(created_at_ms,unit='ms',utc=True,errors='coerce');
                    if pd.isna(game_date): continue
                    variant=game_data.get('variant','standard'); speed=game_data.get('speed','unknown'); perf=game_data.get('perf','unknown'); status=game_data.get('status','unknown'); winner=game_data.get('winner')
                    white_name=white_user.get('name','Unknown'); black_name=black_user.get('name','Unknown'); white_title=white_user.get('title'); black_title=black_user.get('title')
                    white_rating=pd.to_numeric(white_info.get('rating'),errors='coerce'); black_rating=pd.to_numeric(black_info.get('rating'),errors='coerce')
                    player_color,player_elo,opp_name_raw,opp_title_raw,opp_elo=(None,None,'Unknown',None,None)
                    if username_lower==white_name.lower(): player_color,player_elo,opp_name_raw,opp_title_raw,opp_elo=('White',white_rating,black_name,black_title,black_rating)
                    elif username_lower==black_name.lower(): player_color,player_elo,opp_name_raw,opp_title_raw,opp_elo=('Black',black_rating,white_name,white_title,white_rating)
                    else: continue
                    if player_color is None or pd.isna(player_elo) or pd.isna(opp_elo): continue
                    res_num,res_str=(0.5,"Draw");
                    if status not in ['draw','stalemate']:
                       if winner==player_color.lower(): res_num,res_str=(1,"Win")
                       elif winner is not None: res_num,res_str=(0,"Loss")
                    tc_str="Unknown";
                    if clock_info: init=clock_info.get('initial');incr=clock_info.get('increment');
                    if init is not None and incr is not None: tc_str=f"{init}+{incr}"
                    elif speed=='correspondence': tc_str="Correspondence"
                    eco=opening_info.get('eco','Unknown'); op_name_api=opening_info.get('name','Unknown Opening').replace('?','').split(':')[0].strip()
                    op_name_custom=eco_map.get(eco, f"ECO: {eco}" if eco!='Unknown' else 'Unknown Opening')
                    term_map={"mate":"Normal","resign":"Normal","stalemate":"Normal","timeout":"Time forfeit","draw":"Normal","outoftime":"Time forfeit","cheat":"Cheat","noStart":"Aborted","unknownFinish":"Unknown","variantEnd":"Variant End"}
                    term=term_map.get(status,"Unknown")
                    opp_title_final='Unknown'
                    if opp_title_raw and opp_title_raw.strip(): opp_title_clean=opp_title_raw.replace(' ','').strip().upper();
                    if opp_title_clean and opp_title_clean!='?': opp_title_final=opp_title_clean
                    def clean_name(n): return re.sub(r'^(GM|IM|FM|WGM|WIM|WFM|CM|WCM|NM)\s+','',n).strip()
                    opp_name_clean=clean_name(opp_name_raw)
                    all_games_data.append({'Date':game_date,'Event':perf,'White':white_name,'Black':black_name,'Result':"1-0" if winner=='white' else ("0-1" if winner=='black' else "1/2-1/2"),'WhiteElo':int(white_rating) if not pd.isna(white_rating) else 0,'BlackElo':int(black_rating) if not pd.isna(black_rating) else 0,'ECO':eco,'OpeningName_API':op_name_api,'OpeningName_Custom':op_name_custom,'TimeControl':tc_str,'Termination':term,'PlyCount':game_data.get('turns',0),'LichessID':game_id,'PlayerID':username,'PlayerColor':player_color,'PlayerElo':int(player_elo),'OpponentName':opp_name_clean,'OpponentNameRaw':opp_name_raw,'OpponentElo':int(opp_elo),'OpponentTitle':opp_title_final,'PlayerResultNumeric':res_num,'PlayerResultString':res_str,'Variant':variant,'Speed':speed,'Status':status,'PerfType':perf})
                except json.JSONDecodeError: error_counter += 1
                except Exception: error_counter += 1
    except requests.exceptions.RequestException as e: return pd.DataFrame(), f"🚨 API Error: {e}"
    except Exception as e: return pd.DataFrame(), f"🚨 Error: {e}\n{traceback.format_exc()}"
    status_message = f"Processed {len(all_games_data)} games.";
    if error_counter > 0: status_message += f" Skipped {error_counter} errors."
    if not all_games_data: return pd.DataFrame(), f"⚠️ No games found matching criteria."
    if progress: progress(0.8, desc="Finalizing...")
    df = pd.DataFrame(all_games_data);
    if not df.empty:
        df['Date']=pd.to_datetime(df['Date'],errors='coerce'); df=df.dropna(subset=['Date'])
        if df.empty: return df, "⚠️ No games with valid dates."
        df['Year']=df['Date'].dt.year; df['Month']=df['Date'].dt.month; df['Day']=df['Date'].dt.day; df['Hour']=df['Date'].dt.hour; df['DayOfWeekNum']=df['Date'].dt.dayofweek; df['DayOfWeekName']=df['Date'].dt.day_name()
        df['PlayerElo']=df['PlayerElo'].astype(int); df['OpponentElo']=df['OpponentElo'].astype(int)
        df['EloDiff']=df['PlayerElo']-df['OpponentElo']; df['TimeControl_Category']=df.apply(lambda r: categorize_time_control(r['TimeControl'], r['Speed']), axis=1)
        df=df.sort_values(by='Date').reset_index(drop=True)
    if progress: progress(1, desc="Complete!")
    return df, status_message

# =============================================
# Plotting Functions (Unchanged)
# =============================================
# (Insert ALL plotting functions here - code identical to previous version v15)
# ... (plot_win_loss_pie, ..., plot_time_forfeit_by_tc) ...
def plot_win_loss_pie(df, display_name):
    if 'PlayerResultString' not in df.columns: return go.Figure()
    result_counts = df['PlayerResultString'].value_counts()
    fig = px.pie(values=result_counts.values, names=result_counts.index, title=f'Overall Results for {display_name}', color=result_counts.index, color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'}, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05 if x == 'Win' else 0 for x in result_counts.index]); fig.update_layout(dragmode=False); return fig
def plot_win_loss_by_color(df):
    if not all(col in df.columns for col in ['PlayerColor', 'PlayerResultString']): return go.Figure()
    try: color_results=df.groupby(['PlayerColor','PlayerResultString']).size().unstack(fill_value=0)
    except KeyError: return go.Figure().update_layout(title="Error: Missing Columns")
    for res in ['Win','Draw','Loss']: color_results[res]=color_results.get(res,0)
    color_results=color_results[['Win','Draw','Loss']]; total=color_results.sum(axis=1); color_results_pct=color_results.apply(lambda x:x*100/total[x.name] if total[x.name]>0 else 0,axis=1)
    fig=px.bar(color_results_pct, barmode='stack', title='Results by Color', labels={'value':'%', 'PlayerColor':'Played As'}, color='PlayerResultString', color_discrete_map={'Win':'#4CAF50', 'Draw':'#B0BEC5', 'Loss':'#F44336'}, text_auto='.1f', category_orders={"PlayerColor":["White","Black"]})
    fig.update_layout(yaxis_title="Percentage (%)", xaxis_title="Color Played", dragmode=False); fig.update_traces(textangle=0); return fig
def plot_rating_trend(df, display_name):
    if not all(col in df.columns for col in ['Date', 'PlayerElo']): return go.Figure()
    df_plot=df.copy(); df_plot['PlayerElo']=pd.to_numeric(df_plot['PlayerElo'],errors='coerce'); df_sorted=df_plot[df_plot['PlayerElo'].notna() & (df_plot['PlayerElo']>0)].sort_values('Date')
    if df_sorted.empty: return go.Figure().update_layout(title=f"No Elo data")
    fig=go.Figure(); fig.add_trace(go.Scatter(x=df_sorted['Date'], y=df_sorted['PlayerElo'], mode='lines+markers', name='Elo', line=dict(color='#1E88E5',width=2), marker=dict(size=5,opacity=0.7)))
    fig.update_layout(title=f'{display_name}\'s Rating Trend', xaxis_title='Date', yaxis_title='Elo Rating', hovermode="x unified", xaxis_rangeslider_visible=True, dragmode=False); return fig
def plot_performance_vs_opponent_elo(df):
    if not all(col in df.columns for col in ['PlayerResultString', 'EloDiff']): return go.Figure()
    fig=px.box(df, x='PlayerResultString', y='EloDiff', title='Elo Advantage vs. Result', labels={'PlayerResultString':'Result', 'EloDiff':'Your Elo - Opponent Elo'}, category_orders={"PlayerResultString":["Win","Draw","Loss"]}, color='PlayerResultString', color_discrete_map={'Win':'#4CAF50','Draw':'#B0BEC5','Loss':'#F44336'}, points='outliers')
    fig.add_hline(y=0, line_dash="dash", line_color="grey"); fig.update_traces(marker=dict(opacity=0.8)); fig.update_layout(dragmode=False); return fig
def plot_games_by_dow(df):
    if 'DayOfWeekName' not in df.columns: return go.Figure()
    dow_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    games_by_dow=df['DayOfWeekName'].value_counts().reindex(dow_order, fill_value=0)
    fig=px.bar(games_by_dow, x=games_by_dow.index, y=games_by_dow.values, title="Games by Day of Week", labels={'x':'Day','y':'Games'}, text=games_by_dow.values)
    fig.update_traces(marker_color='#9C27B0', textposition='outside'); fig.update_layout(dragmode=False); return fig
def plot_winrate_by_dow(df):
    if not all(col in df.columns for col in ['DayOfWeekName', 'PlayerResultNumeric']): return go.Figure()
    dow_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wins_by_dow=df[df['PlayerResultNumeric']==1].groupby('DayOfWeekName').size(); total_by_dow=df.groupby('DayOfWeekName').size()
    win_rate=(wins_by_dow.reindex(total_by_dow.index,fill_value=0)/total_by_dow).fillna(0)*100
    win_rate=win_rate.reindex(dow_order,fill_value=0)
    fig=px.bar(win_rate, x=win_rate.index, y=win_rate.values, title="Win Rate (%) by Day", labels={'x':'Day','y':'Win Rate (%)'}, text=win_rate.values)
    fig.update_traces(marker_color='#FF9800', texttemplate='%{text:.1f}%', textposition='outside'); fig.update_layout(yaxis_range=[0,100], dragmode=False); return fig
def plot_games_by_hour(df):
    if 'Hour' not in df.columns: return go.Figure()
    games_by_hour=df['Hour'].value_counts().sort_index().reindex(range(24),fill_value=0)
    fig=px.bar(games_by_hour, x=games_by_hour.index, y=games_by_hour.values, title="Games by Hour (UTC)", labels={'x':'Hour','y':'Games'}, text=games_by_hour.values)
    fig.update_traces(marker_color='#03A9F4', textposition='outside'); fig.update_layout(xaxis=dict(tickmode='linear'), dragmode=False); return fig
def plot_winrate_by_hour(df):
    if not all(col in df.columns for col in ['Hour', 'PlayerResultNumeric']): return go.Figure()
    wins_by_hour=df[df['PlayerResultNumeric']==1].groupby('Hour').size(); total_by_hour=df.groupby('Hour').size()
    win_rate=(wins_by_hour.reindex(total_by_hour.index,fill_value=0)/total_by_hour).fillna(0)*100
    win_rate=win_rate.reindex(range(24),fill_value=0)
    fig=px.line(win_rate, x=win_rate.index, y=win_rate.values, markers=True, title="Win Rate (%) by Hour (UTC)", labels={'x':'Hour','y':'Win Rate (%)'})
    fig.update_traces(line_color='#8BC34A'); fig.update_layout(yaxis_range=[0,100], xaxis=dict(tickmode='linear'), dragmode=False); return fig
def plot_games_per_year(df):
    if 'Year' not in df.columns: return go.Figure()
    games_per_year = df['Year'].value_counts().sort_index()
    fig = px.bar(games_per_year, x=games_per_year.index, y=games_per_year.values, title='Games Per Year', labels={'x':'Year','y':'Games'}, text=games_per_year.values)
    fig.update_traces(marker_color='#2196F3', textposition='outside'); fig.update_layout(xaxis_title="Year", yaxis_title="Number of Games", xaxis={'type':'category'}, dragmode=False); return fig
def plot_win_rate_per_year(df):
    if not all(col in df.columns for col in ['Year', 'PlayerResultNumeric']): return go.Figure()
    wins_per_year=df[df['PlayerResultNumeric']==1].groupby('Year').size(); total_per_year=df.groupby('Year').size()
    win_rate=(wins_per_year.reindex(total_per_year.index,fill_value=0)/total_per_year).fillna(0)*100
    win_rate.index=win_rate.index.astype(str)
    fig=px.line(win_rate, x=win_rate.index, y=win_rate.values, title='Win Rate (%) Per Year', markers=True, labels={'x':'Year','y':'Win Rate (%)'})
    fig.update_traces(line_color='#FFC107', line_width=2.5); fig.update_layout(yaxis_range=[0,100], dragmode=False); return fig
def plot_performance_by_time_control(df):
     if not all(col in df.columns for col in ['TimeControl_Category', 'PlayerResultString']): return go.Figure()
     try:
        tc_results=df.groupby(['TimeControl_Category','PlayerResultString']).size().unstack(fill_value=0)
        for res in ['Win','Draw','Loss']: tc_results[res]=tc_results.get(res,0)
        tc_results=tc_results[['Win','Draw','Loss']]; total=tc_results.sum(axis=1)
        tc_results_pct=tc_results.apply(lambda x:x*100/total[x.name] if total[x.name]>0 else 0, axis=1)
        found=df['TimeControl_Category'].unique(); pref=['Bullet','Blitz','Rapid','Classical','Correspondence','Unknown']
        order=[c for c in pref if c in found]+[c for c in found if c not in pref]
        tc_results_pct=tc_results_pct.reindex(index=order).dropna(axis=0,how='all')
        fig=px.bar(tc_results_pct, title='Performance by Time Control', labels={'value':'%','TimeControl_Category':'Category'}, color='PlayerResultString', color_discrete_map={'Win':'#4CAF50','Draw':'#B0BEC5','Loss':'#F44336'}, barmode='group', text_auto='.1f')
        fig.update_layout(xaxis_title="Time Control Category", yaxis_title="Percentage (%)", dragmode=False); fig.update_traces(textangle=0); return fig
     except Exception: return go.Figure().update_layout(title="Error")
def plot_opening_frequency(df, top_n=20, opening_col='OpeningName_API'):
    if opening_col not in df.columns: return go.Figure()
    source_label = "Lichess API" if opening_col == 'OpeningName_API' else "Custom Mapping"
    opening_counts = df[df[opening_col] != 'Unknown Opening'][opening_col].value_counts().nlargest(top_n)
    fig = px.bar(opening_counts, y=opening_counts.index, x=opening_counts.values, orientation='h', title=f'Top {top_n} Openings ({source_label})', labels={'y':'Opening','x':'Games'}, text=opening_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, dragmode=False); fig.update_traces(marker_color='#673AB7', textposition='outside'); return fig
def plot_win_rate_by_opening(df, min_games=5, top_n=20, opening_col='OpeningName_API'):
    if not all(col in df.columns for col in [opening_col, 'PlayerResultNumeric']): return go.Figure()
    source_label = "Lichess API" if opening_col == 'OpeningName_API' else "Custom Mapping"
    opening_stats=df.groupby(opening_col).agg(total_games=('PlayerResultNumeric','count'), wins=('PlayerResultNumeric',lambda x:(x==1).sum()))
    opening_stats=opening_stats[(opening_stats['total_games']>=min_games)&(opening_stats.index!='Unknown Opening')].copy()
    if opening_stats.empty: return go.Figure().update_layout(title=f"No openings >= {min_games} games ({source_label})")
    opening_stats['win_rate']=(opening_stats['wins']/opening_stats['total_games'])*100
    opening_stats_plot=opening_stats.nlargest(top_n, 'win_rate')
    fig=px.bar(opening_stats_plot, y=opening_stats_plot.index, x='win_rate', orientation='h', title=f'Top {top_n} Openings by Win Rate (Min {min_games} games, {source_label})', labels={'win_rate':'Win Rate (%)',opening_col:'Opening'}, text='win_rate')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside', marker_color='#009688'); fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Win Rate (%)", dragmode=False); return fig
def plot_most_frequent_opponents(df, top_n=20):
    if 'OpponentName' not in df.columns: return go.Figure()
    opp_counts=df[df['OpponentName']!='Unknown']['OpponentName'].value_counts().nlargest(top_n)
    fig=px.bar(opp_counts, y=opp_counts.index, x=opp_counts.values, orientation='h', title=f'Top {top_n} Opponents', labels={'y':'Opponent','x':'Games'}, text=opp_counts.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, dragmode=False); fig.update_traces(marker_color='#FF5722', textposition='outside'); return fig
def plot_games_by_dom(df):
    if 'Day' not in df.columns: return go.Figure()
    games_by_dom = df['Day'].value_counts().sort_index().reindex(range(1, 32), fill_value=0)
    fig = px.bar(games_by_dom, x=games_by_dom.index, y=games_by_dom.values, title="Games Played per Day of Month", labels={'x': 'Day of Month', 'y': 'Number of Games'}, text=games_by_dom.values)
    fig.update_traces(marker_color='#E91E63', textposition='outside'); fig.update_layout(xaxis=dict(tickmode='linear'), dragmode=False); return fig
def plot_winrate_by_dom(df):
    if not all(col in df.columns for col in ['Day', 'PlayerResultNumeric']): return go.Figure()
    wins_by_dom=df[df['PlayerResultNumeric']==1].groupby('Day').size(); total_by_dom=df.groupby('Day').size()
    win_rate=(wins_by_dom.reindex(total_by_dom.index,fill_value=0)/total_by_dom).fillna(0)*100
    win_rate=win_rate.reindex(range(1,32),fill_value=0)
    fig=px.line(win_rate, x=win_rate.index, y=win_rate.values, markers=True, title="Win Rate (%) per Day of Month", labels={'x': 'Day of Month', 'y': 'Win Rate (%)'})
    fig.update_traces(line_color='#FF5722'); fig.update_layout(yaxis_range=[0,100], xaxis=dict(tickmode='linear'), dragmode=False); return fig
def plot_time_forfeit_summary(wins_tf, losses_tf):
    data={'Outcome':['Won on Time','Lost on Time'],'Count':[wins_tf,losses_tf]}
    df_tf=pd.DataFrame(data)
    fig=px.bar(df_tf,x='Outcome',y='Count',title="Time Forfeit Summary", color='Outcome', color_discrete_map={'Won on Time':'#4CAF50','Lost on Time':'#F44336'}, text='Count')
    fig.update_layout(showlegend=False, dragmode=False); fig.update_traces(textposition='outside'); return fig
def plot_time_forfeit_by_tc(tf_games_df):
    if 'TimeControl_Category' not in tf_games_df.columns or tf_games_df.empty: return go.Figure().update_layout(title="No TF Data by Category")
    tf_by_tc=tf_games_df['TimeControl_Category'].value_counts()
    fig=px.bar(tf_by_tc,x=tf_by_tc.index,y=tf_by_tc.values, title="Time Forfeits by Time Control", labels={'x':'Category','y':'Forfeits'}, text=tf_by_tc.values)
    fig.update_layout(dragmode=False); fig.update_traces(marker_color='#795548', textposition='outside'); return fig

# =============================================
# Helper Functions
# =============================================
# ... (Functions identical to v15) ...
def filter_and_analyze_titled(df, titles):
    if 'OpponentTitle' not in df.columns: return pd.DataFrame()
    titled_games = df[df['OpponentTitle'].isin(titles)].copy(); return titled_games
def filter_and_analyze_time_forfeits(df):
    if 'Termination' not in df.columns: return pd.DataFrame(), 0, 0
    tf_games = df[df['Termination'].str.contains("Time forfeit", na=False, case=False)].copy()
    if tf_games.empty: return tf_games, 0, 0
    wins_tf = len(tf_games[tf_games['PlayerResultNumeric'] == 1])
    losses_tf = len(tf_games[tf_games['PlayerResultNumeric'] == 0])
    return tf_games, wins_tf, losses_tf

# =============================================
# Gradio Main Analysis Function (Correct)
# =============================================
# ... (Function identical to v15) ...
def perform_full_analysis(username, time_period_key, perf_type, selected_titles_list, progress=gr.Progress(track_tqdm=True)): # Added track_tqdm back
    df, status_msg = load_from_lichess_api(username, time_period_key, perf_type, DEFAULT_RATED_ONLY, ECO_MAPPING, progress)
    num_outputs = 30 # Recalculate based on the exact number of output components below
    if not isinstance(df, pd.DataFrame) or df.empty:
        return status_msg, pd.DataFrame(), *( [None] * (num_outputs - 2) ) # Return Nones for plot/df components
    try:
        # Generate all plots and data...
        fig_pie=plot_win_loss_pie(df,username); fig_color=plot_win_loss_by_color(df); fig_rating=plot_rating_trend(df,username); fig_elo_diff=plot_performance_vs_opponent_elo(df)
        total_g=len(df); w=len(df[df['PlayerResultNumeric']==1]); l=len(df[df['PlayerResultNumeric']==0]); d=len(df[df['PlayerResultNumeric']==0.5])
        wr=(w/total_g*100) if total_g>0 else 0; avg_opp=df['OpponentElo'].mean(); overview_stats_md=f"**Total:** {total_g:,} | **WR:** {wr:.1f}% | **W/L/D:** {w}/{l}/{d} | **Avg Opp:** {avg_opp:.0f if not pd.isna(avg_opp) else 'N/A'}"
        fig_games_yr=plot_games_per_year(df); fig_wr_yr=plot_win_rate_per_year(df); fig_perf_tc=plot_performance_by_time_control(df)
        fig_games_dow=plot_games_by_dow(df); fig_wr_dow=plot_winrate_by_dow(df); fig_games_hod=plot_games_by_hour(df); fig_wr_hod=plot_winrate_by_hour(df)
        fig_games_dom=plot_games_by_dom(df); fig_wr_dom=plot_winrate_by_dom(df)
        fig_open_freq_api=plot_opening_frequency(df,top_n=15,opening_col='OpeningName_API'); fig_open_wr_api=plot_win_rate_by_opening(df,min_games=5,top_n=15,opening_col='OpeningName_API')
        fig_open_freq_cust=plot_opening_frequency(df,top_n=15,opening_col='OpeningName_Custom') if ECO_MAPPING else go.Figure().update_layout(title="Custom Map Unavailable"); fig_open_wr_cust=plot_win_rate_by_opening(df,min_games=5,top_n=15,opening_col='OpeningName_Custom') if ECO_MAPPING else go.Figure().update_layout(title="Custom Map Unavailable")
        fig_opp_freq=plot_most_frequent_opponents(df,top_n=20); df_opp_list=df[df['OpponentName']!='Unknown']['OpponentName'].value_counts().reset_index(name='Games').head(20) if 'OpponentName' in df else pd.DataFrame(); fig_opp_elo=plot_performance_vs_opponent_elo(df)
        tf_games,wins_tf,losses_tf=filter_and_analyze_time_forfeits(df)
        fig_tf_summary=plot_time_forfeit_summary(wins_tf,losses_tf) if not tf_games.empty else go.Figure().update_layout(title="No Time Forfeit Data"); fig_tf_tc=plot_time_forfeit_by_tc(tf_games) if not tf_games.empty else go.Figure().update_layout(title="No TF Data by Category")
        df_tf_list=tf_games[['Date','OpponentName','PlayerColor','PlayerResultString','TimeControl','PlyCount','Termination']].sort_values('Date',ascending=False).head(20) if not tf_games.empty else pd.DataFrame()
        term_counts=df['Termination'].value_counts(); fig_term_all=px.bar(term_counts,x=term_counts.index,y=term_counts.values,title="Overall Termination Reasons",labels={'x':'Reason','y':'Count'},text=term_counts.values)
        fig_term_all.update_layout(dragmode=False); fig_term_all.update_traces(textposition='outside')
        # Generate Titled Player analysis...
        titled_status_msg = ""; fig_titled_pie, fig_titled_color, fig_titled_rating, df_titled_h2h = go.Figure(), go.Figure(), go.Figure(), pd.DataFrame()
        if selected_titles_list:
            titled_games = filter_and_analyze_titled(df, selected_titles_list)
            if not titled_games.empty:
                titled_status_msg = f"✅ Found {len(titled_games)} games vs {', '.join(selected_titles_list)}."
                fig_titled_pie = plot_win_loss_pie(titled_games, f"{username} vs Titles")
                fig_titled_color = plot_win_loss_by_color(titled_games)
                fig_titled_rating = plot_rating_trend(titled_games, f"{username} (vs Titles)")
                h2h = titled_games.groupby('OpponentNameRaw')['PlayerResultString'].value_counts().unstack(fill_value=0)
                for res in ['Win','Loss','Draw']: h2h[res]=h2h.get(res,0)
                h2h = h2h[['Win','Loss','Draw']]; h2h['Total']=h2h.sum(axis=1); h2h['Score']=h2h['Win']+0.5*h2h['Draw']
                df_titled_h2h = h2h.sort_values('Total', ascending=False).reset_index()
            else: titled_status_msg = f"ℹ️ No games found vs selected titles ({', '.join(selected_titles_list)})."
        else: titled_status_msg = "ℹ️ Select titles from the sidebar to analyze."
        # Return all results... MUST match outputs_list order
        # Recalculate num_outputs based on this exact return statement
        return_tuple = ( status_msg, df, fig_pie, overview_stats_md, fig_color, fig_rating, fig_elo_diff, fig_games_yr, fig_wr_yr, "(Results by color shown in Overview)", fig_games_dow, fig_wr_dow, fig_games_hod, fig_wr_hod, fig_games_dom, fig_wr_dom, fig_perf_tc, fig_open_freq_api, fig_open_wr_api, fig_open_freq_cust, fig_open_wr_cust, fig_opp_freq, df_opp_list, fig_opp_elo, titled_status_msg, fig_titled_pie, fig_titled_color, fig_titled_rating, df_titled_h2h, fig_tf_summary, fig_tf_tc, df_tf_list, fig_term_all )
        # print(f"DEBUG: Returning {len(return_tuple)} items from perform_full_analysis") # Check length if needed
        return return_tuple
    except Exception as e:
        error_msg = f"🚨 Error generating results: {e}\n{traceback.format_exc()}";
        num_outputs = 32 # Updated count based on return tuple above
        return error_msg, pd.DataFrame(), *( [None] * (num_outputs - 2) )


# =============================================
# Gradio Interface Definition (Corrected UI Syntax)
# =============================================
css = """.gradio-container { font-family: 'IBM Plex Sans', sans-serif; } footer { display: none !important; }"""
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# ♟️ Lichess Insights\nAnalyze rated game statistics from Lichess API.")
    df_state = gr.State(pd.DataFrame())

    with gr.Row():
        with gr.Column(scale=1, min_width=250): # Sidebar
            gr.Markdown("## ⚙️ Settings"); username_input=gr.Textbox(label="Lichess Username", placeholder="e.g., DrNykterstein", elem_id="username_box"); time_period_input=gr.Dropdown(label="Time Period", choices=list(TIME_PERIOD_OPTIONS.keys()), value=DEFAULT_TIME_PERIOD); perf_type_input=gr.Dropdown(label="Game Type", choices=PERF_TYPE_OPTIONS_SINGLE, value=DEFAULT_PERF_TYPE); analyze_btn=gr.Button("Analyze Games", variant="primary"); status_output=gr.Markdown(""); gr.Markdown("---"); gr.Markdown("### Analyze vs Titled Players"); titled_player_select=gr.CheckboxGroup(label="Select Opponent Titles", choices=TITLES_TO_ANALYZE, value=['GM', 'IM'], elem_id="titled_select"); gr.Markdown("*(Analysis updates on 'Analyze Games' click)*");
        with gr.Column(scale=4): # Main Content
            # Define Output Components - Order Matters! Match return tuple exactly.
            overview_plot_pie=gr.Plot(label="Overall Results"); overview_stats_md_out=gr.Markdown(); overview_plot_color=gr.Plot(label="Results by Color"); overview_plot_rating=gr.Plot(label="Rating Trend"); overview_plot_elo_diff=gr.Plot(label="Elo Advantage vs. Result")
            time_plot_games_yr=gr.Plot(label="Games per Year"); time_plot_wr_yr=gr.Plot(label="Win Rate per Year")
            color_plot_placeholder=gr.Markdown()
            time_plot_games_dow=gr.Plot(label="Games by Day of Week"); time_plot_wr_dow=gr.Plot(label="Win Rate by Day of Week"); time_plot_games_hod=gr.Plot(label="Games by Hour (UTC)"); time_plot_wr_hod=gr.Plot(label="Win Rate by Hour (UTC)"); time_plot_games_dom=gr.Plot(label="Games by Day of Month"); time_plot_wr_dom=gr.Plot(label="Win Rate by Day of Month"); time_plot_perf_tc=gr.Plot(label="Performance by Time Control")
            eco_plot_freq_api=gr.Plot(label="Opening Frequency (API)"); eco_plot_wr_api=gr.Plot(label="Opening Win Rate (API)"); eco_plot_freq_cust=gr.Plot(label="Opening Frequency (Custom)"); eco_plot_wr_cust=gr.Plot(label="Opening Win Rate (Custom)")
            opp_plot_freq=gr.Plot(label="Frequent Opponents"); opp_df_list=gr.DataFrame(label="Top Opponents List", wrap=True); opp_plot_elo=gr.Plot(label="Elo Advantage vs Result")
            titled_status=gr.Markdown(); titled_plot_pie=gr.Plot(label="Results vs Selected Titles"); titled_plot_color=gr.Plot(label="Results by Color vs Selected Titles"); titled_plot_rating=gr.Plot(label="Rating Trend vs Selected Titles"); titled_df_h2h_comp=gr.DataFrame(label="Head-to-Head vs Selected Titles", wrap=True); # Component name
            term_plot_tf_summary=gr.Plot(label="Time Forfeit Summary"); term_plot_tf_tc=gr.Plot(label="Time Forfeits by Time Control"); term_df_tf_list=gr.DataFrame(label="Recent TF Games", wrap=True); term_plot_all=gr.Plot(label="Overall Termination")

            # Arrange Components in Tabs - Using CORRECT block structure
            with gr.Tabs() as tabs:
                with gr.TabItem("1. Overview", id=0):
                    overview_stats_md_out # Display metrics
                    with gr.Row():
                        overview_plot_pie
                        overview_plot_color
                    overview_plot_rating
                    overview_plot_elo_diff

                with gr.TabItem("2. Perf. Over Time", id=1):
                    # Note: Rating Trend is defined in Overview, reference it here if needed
                    # For clarity, let's just place the year-based plots
                    time_plot_games_yr
                    time_plot_wr_yr

                with gr.TabItem("3. Perf. by Color", id=2):
                    overview_plot_color # Reuse color plot
                    color_plot_placeholder # Display placeholder text

                with gr.TabItem("4. Time & Date", id=3):
                    gr.Markdown("### Day of Week")
                    with gr.Row():
                        time_plot_games_dow
                        time_plot_wr_dow
                    gr.Markdown("### Hour of Day (UTC)")
                    with gr.Row():
                        time_plot_games_hod
                        time_plot_wr_hod
                    gr.Markdown("### Day of Month")
                    with gr.Row():
                        time_plot_games_dom
                        time_plot_wr_dom
                    gr.Markdown("### Time Control Category")
                    time_plot_perf_tc

                with gr.TabItem("5. ECO & Openings", id=4):
                    gr.Markdown("#### Based on Lichess API Opening Names")
                    eco_plot_freq_api
                    eco_plot_wr_api
                    gr.Markdown("---")
                    gr.Markdown("#### Based on Custom ECO Map")
                    if not ECO_MAPPING:
                        gr.Markdown("⚠️ Custom ECO map file not loaded.")
                    else:
                        eco_plot_freq_cust
                        eco_plot_wr_cust

                with gr.TabItem("6. Opponents", id=5):
                    opp_plot_freq
                    opp_df_list
                    opp_plot_elo

                with gr.TabItem("7. vs Titled", id=6):
                    gr.Markdown("Analysis based on titles selected in the sidebar.")
                    titled_status # Show status message
                    with gr.Row():
                        titled_plot_pie
                        titled_plot_color
                    titled_plot_rating
                    titled_df_h2h_comp # Show H2H table using the component

                with gr.TabItem("8. Termination", id=7):
                    gr.Markdown("### Time Forfeit")
                    term_plot_tf_summary
                    term_plot_tf_tc
                    with gr.Accordion("View Recent TF Games", open=False):
                        term_df_tf_list
                    gr.Markdown("### Overall Termination")
                    term_plot_all

    # Define the list of output components in the exact order
    outputs_list = [
        status_output, df_state, # Status and State first
        overview_plot_pie, overview_stats_md_out, overview_plot_color, overview_plot_rating, overview_plot_elo_diff, # Tab 1
        time_plot_games_yr, time_plot_wr_yr, # Tab 2
        color_plot_placeholder, # Tab 3
        time_plot_games_dow, time_plot_wr_dow, time_plot_games_hod, time_plot_wr_hod, time_plot_games_dom, time_plot_wr_dom, time_plot_perf_tc, # Tab 4
        eco_plot_freq_api, eco_plot_wr_api, eco_plot_freq_cust, eco_plot_wr_cust, # Tab 5
        opp_plot_freq, opp_df_list, opp_plot_elo, # Tab 6
        titled_status, titled_plot_pie, titled_plot_color, titled_plot_rating, titled_df_h2h_comp, # Tab 7 (Correct component name)
        term_plot_tf_summary, term_plot_tf_tc, term_df_tf_list, term_plot_all # Tab 8
    ]
    # Calculate length dynamically to avoid manual count errors
    expected_outputs_count = len(outputs_list)
    print(f"DEBUG: Expected number of outputs for Gradio: {expected_outputs_count}") # Log this!

    # Modify the perform_full_analysis return tuple logic for errors
    def perform_full_analysis_wrapper(username, time_period_key, perf_type, selected_titles_list, progress=gr.Progress(track_tqdm=True)):
         results = perform_full_analysis(username, time_period_key, perf_type, selected_titles_list, progress)
         # Ensure the correct number of outputs is always returned, even on error
         if len(results) != expected_outputs_count:
              print(f"WARN: Mismatch in expected ({expected_outputs_count}) vs actual ({len(results)}) outputs!")
              # Pad with Nones if too few, or truncate if too many (though padding is safer)
              if len(results) < expected_outputs_count:
                   results = tuple(list(results) + [None] * (expected_outputs_count - len(results)))
              else:
                   results = results[:expected_outputs_count]
         return results

    # Connect button click to the wrapper function
    analyze_btn.click(
        fn=perform_full_analysis_wrapper, # Use the wrapper
        inputs=[username_input, time_period_input, perf_type_input, titled_player_select],
        outputs=outputs_list
    )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    demo.launch(debug=True) # Keep debug True for local testing
