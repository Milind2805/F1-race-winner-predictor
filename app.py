from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import shap
import plotly.graph_objects as go
import json
# ── 2023 Championship order (default grid) ─────────────────────
default_grid_2023 = {
    1:  {'driver': 'VER', 'team': 'Red Bull Racing', 'points': 575},
    2:  {'driver': 'PER', 'team': 'Red Bull Racing', 'points': 285},
    3:  {'driver': 'ALO', 'team': 'Aston Martin',    'points': 206},
    4:  {'driver': 'HAM', 'team': 'Mercedes',        'points': 234},
    5:  {'driver': 'SAI', 'team': 'Ferrari',         'points': 200},
    6:  {'driver': 'LEC', 'team': 'Ferrari',         'points': 206},
    7:  {'driver': 'NOR', 'team': 'McLaren',         'points': 205},
    8:  {'driver': 'RUS', 'team': 'Mercedes',        'points': 175},
    9:  {'driver': 'PIA', 'team': 'McLaren',         'points': 97},
    10: {'driver': 'STR', 'team': 'Aston Martin',    'points': 74},
    11: {'driver': 'GAS', 'team': 'Alpine',          'points': 62},
    12: {'driver': 'OCO', 'team': 'Alpine',          'points': 58},
    13: {'driver': 'ALB', 'team': 'Williams',        'points': 27},
    14: {'driver': 'TSU', 'team': 'AlphaTauri',      'points': 17},
    15: {'driver': 'BOT', 'team': 'Alfa Romeo',      'points': 10},
    16: {'driver': 'HUL', 'team': 'Haas F1 Team',   'points': 9},
    17: {'driver': 'ZHO', 'team': 'Alfa Romeo',      'points': 6},
    18: {'driver': 'MAG', 'team': 'Haas F1 Team',   'points': 3},
    19: {'driver': 'RIC', 'team': 'AlphaTauri',      'points': 6},
    20: {'driver': 'SAR', 'team': 'Williams',        'points': 1},
}

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Winner Predictor",
    page_icon="🏎️",
    layout="wide"
)
CURRENT_DIR=Path(__file__).parent

# ── Load model, scaler, encoders, data ───────────────────────
@st.cache_resource
def load_models():
    with open(CURRENT_DIR / 'f1_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(CURRENT_DIR / 'f1_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(CURRENT_DIR / 'f1_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

@st.cache_data
def load_data():
    return pd.read_csv(CURRENT_DIR / 'f1_multiyear.csv')
import json

# Add to your load functions
@st.cache_data
def load_feature_importance():
    with open(CURRENT_DIR / 'f1_feature_importance.json', 'r') as f:
        return json.load(f)

importance_data = load_feature_importance()

model, scaler, encoders = load_models()
df = load_data()
@st.cache_data
def load_shap_values():
    with open(CURRENT_DIR / 'f1_shap_values.json', 'r') as f:
        return json.load(f)

shap_data = load_shap_values()

# ── Title ─────────────────────────────────────────────────────
st.title("🏎️ F1 Race Winner Predictor")
st.markdown("### Predict who wins based on qualifying, weather & standings")
st.divider()

# ── Sidebar — User Inputs ─────────────────────────────────────
st.sidebar.header("🔧 Race Settings")

drivers = sorted(encoders['driver'].classes_.tolist())
teams = sorted(encoders['team'].classes_.tolist())
circuits = sorted(encoders['circuit'].classes_.tolist())

selected_driver = st.sidebar.selectbox("Select Driver", drivers)
selected_team = st.sidebar.selectbox("Select Team", teams)
selected_circuit = st.sidebar.selectbox("Select Circuit", circuits)
grid_position = st.sidebar.slider("Grid Position", 1, 20, 1)
quali_position = st.sidebar.slider("Qualifying Position", 1, 20, 1)
standing_points = st.sidebar.number_input(
    "Driver's Championship Points (before race)", 
    min_value=0, max_value=500, value=100
)
rainfall = st.sidebar.toggle("🌧️ Wet Race?")
air_temp = st.sidebar.slider("Air Temperature (°C)", 10, 50, 25)
track_temp = st.sidebar.slider("Track Temperature (°C)", 15, 70, 35)
home_race = st.sidebar.toggle("🏠 Home Race?")

# ── Predict button ────────────────────────────────────────────
st.sidebar.divider()
predict_btn = st.sidebar.button("🏁 Predict Winner", use_container_width=True)

# ── Main area — tabs ──────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏁 Prediction",
    "📊 Season Analysis",
    "🏆 Constructor Stats",
    "🔍 Model Insights",
    "⚔️ Head-to-Head",
    "🧠 SHAP Explainer",
    "🎮 Race Simulation"
])
# ════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════
with tab1:
    if predict_btn:
        # Encode inputs
        try:
            driver_enc = encoders['driver'].transform([selected_driver])[0]
            team_enc = encoders['team'].transform([selected_team])[0]
            circuit_enc = encoders['circuit'].transform([selected_circuit])[0]
        except ValueError as e:
            st.error(f"Encoding error: {e}")
            st.stop()

        input_data = np.array([[
            driver_enc, team_enc, circuit_enc,
            grid_position, quali_position,
            int(rainfall), air_temp, track_temp,
            standing_points, int(home_race)
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Show result
        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.success(f"🏆 {selected_driver} is predicted to WIN!")
            else:
                st.error(f"❌ {selected_driver} is NOT predicted to win")

        with col2:
            st.metric("Win Probability", f"{probability*100:.1f}%")

        with col3:
            st.metric("Grid Position", f"P{grid_position}")

        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            title={'text': "Win Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e8002d"},
                'steps': [
                    {'range': [0, 30], 'color': "#1a1a2e"},
                    {'range': [30, 60], 'color': "#16213e"},
                    {'range': [60, 100], 'color': "#0f3460"},
                ]
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=300
        )
        st.plotly_chart(fig,width='stretch')

    else:
        st.info("👈 Set race parameters in the sidebar and click **Predict Winner**")

# ════════════════════════════════════════════════
# TAB 2 — SEASON ANALYSIS
# ════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Driver Points Progression")

    selected_year = st.selectbox("Select Season", [2019, 2020, 2021, 2022, 2023])
    season_df = df[df['Year'] == selected_year].copy()

    # Top 5 drivers by total points
    top5 = season_df.groupby('Abbreviation')['Points'].sum().nlargest(5).index.tolist()
    season_top = season_df[season_df['Abbreviation'].isin(top5)]
    races_2019=['Australian Grand Prix', 'Bahrain Grand Prix', 'Chinese Grand Prix', 'Azerbaijan Grand Prix', 'Spanish Grand Prix', 'Monaco Grand Prix', 'Canadian Grand Prix', 'French Grand Prix', 'Austrian Grand Prix', 'British Grand Prix', 'German Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 'Italian Grand Prix', 'Singapore Grand Prix', 'Russian Grand Prix', 'Japanese Grand Prix', 
            'Mexican Grand Prix', 'United States Grand Prix', 'Brazilian Grand Prix', 'Abu Dhabi Grand Prix']
    races_2020=['Austrian Grand Prix', 'Styrian Grand Prix', 'Hungarian Grand Prix', 'British Grand Prix', '70th Anniversary Grand Prix', 'Spanish Grand Prix', 'Belgian Grand Prix', 'Italian Grand Prix', 'Tuscan Grand Prix', 
            'Russian Grand Prix', 'Eifel Grand Prix', 'Portuguese Grand Prix', 'Emilia Romagna Grand Prix', 'Turkish Grand Prix', 'Bahrain Grand Prix', 'Sakhir Grand Prix', 'Abu Dhabi Grand Prix']
    races_2021=['Bahrain Grand Prix', 'Emilia Romagna Grand Prix', 'Portuguese Grand Prix', 'Spanish Grand Prix', 'Monaco Grand Prix', 'Azerbaijan Grand Prix', 'French Grand Prix', 'Styrian Grand Prix', 'Austrian Grand Prix', 'British Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 'Italian Grand Prix',
            'Russian Grand Prix', 'Turkish Grand Prix', 'United States Grand Prix', 'Mexico City Grand Prix', 'Brazilian Grand Prix', 'Qatar Grand Prix', 'Saudi Arabian Grand Prix', 'Abu Dhabi Grand Prix']
    races_2022=['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Emilia Romagna Grand Prix', 'Miami Grand Prix', 'Spanish Grand Prix', 'Monaco Grand Prix', 'Azerbaijan Grand Prix', 'Canadian Grand Prix', 'British Grand Prix', 'Austrian Grand Prix', 'French Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 'Italian Grand Prix', 'Singapore Grand Prix', 'Japanese Grand Prix',
            'United States Grand Prix', 'Mexico City Grand Prix', 'Brazilian Grand Prix', 'Abu Dhabi Grand Prix']
    races_2023=['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix', 'Azerbaijan Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix', 'Spanish Grand Prix', 'Canadian Grand Prix', 'Austrian Grand Prix', 'British Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 'Italian Grand Prix', 'Singapore Grand Prix', 'Japanese Grand Prix',
            'Qatar Grand Prix', 'United States Grand Prix', 'Mexico City Grand Prix', 'Brazilian Grand Prix', 'Las Vegas Grand Prix', 'Abu Dhabi Grand Prix']
    all_seasons={2019: races_2019, 2020: races_2020, 2021: races_2021, 2022: races_2022, 2023: races_2023}
    race_order = list(all_seasons[selected_year])

    # Cumulative points per driver
    fig2 = px.line(
        season_top.sort_values('Race'),
        x='Race', y='Points',
        color='Abbreviation',
        title=f"Top 5 Drivers — {selected_year} Points Progression",
        markers=True,
        category_orders={'Race': race_order}
    )
    fig2.update_layout(
        xaxis_tickangle=-45,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Race wins bar chart
    st.subheader("🏆 Race Wins by Driver")
    winners = season_df[season_df['Position'] == 1]['Abbreviation'].value_counts().reset_index()
    winners.columns = ['Driver', 'Wins']
    fig3 = px.bar(
        winners, x='Wins', y='Driver',
        orientation='h',
        color='Wins',
        color_continuous_scale='Reds',
        title=f"Race Wins — {selected_year}"
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 3 — CONSTRUCTOR STATS
# ════════════════════════════════════════════════
with tab3:
    st.subheader("🏗️ Constructor Championship")

    selected_year2 = st.selectbox(
        "Select Season", [2019, 2020, 2021, 2022, 2023], key='constructor_year'
    )
    season_df2 = df[df['Year'] == selected_year2].copy()

    # Constructor points
    constructor_pts = season_df2.groupby('TeamName')['Points'].sum().reset_index()
    constructor_pts.columns = ['Team', 'Points']
    constructor_pts = constructor_pts.sort_values('Points', ascending=True)

    fig4 = px.bar(
        constructor_pts, x='Points', y='Team',
        orientation='h',
        color='Points',
        color_continuous_scale='Blues',
        title=f"Constructor Points — {selected_year2}"
    )
    fig4.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Constructor wins
    st.subheader("🏆 Constructor Wins")
    team_wins = season_df2[season_df2['Position'] == 1]['TeamName'].value_counts().reset_index()
    team_wins.columns = ['Team', 'Wins']
    fig5 = px.bar(
        team_wins, x='Wins', y='Team',
        orientation='h',
        color='Wins',
        color_continuous_scale='Greens',
        title=f"Constructor Wins — {selected_year2}"
    )
    fig5.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.subheader("🔍 What Drives the Model's Predictions?")
    
    imp_df = pd.DataFrame({
        'Feature': list(importance_data.keys()),
        'Importance': list(importance_data.values())
    }).sort_values('Importance', ascending=True)
    
    fig6 = px.bar(
        imp_df, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Oranges',
        title="XGBoost Feature Importance"
    )
    fig6.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400
    )
    st.plotly_chart(fig6, width='stretch')
    
    st.markdown("""
    **How to read this:** Higher values mean the model relies on that feature more 
    heavily when predicting a race winner. This reflects patterns learned from 
    2019-2023 race data.
    """)
with tab5:
    st.subheader("⚔️ Compare Two Drivers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Driver A")
        driver_a = st.selectbox("Select Driver A", drivers, key='driver_a')
        team_a = st.selectbox("Select Team A", teams, key='team_a')
        grid_a = st.slider("Grid Position A", 1, 20, 1, key='grid_a')
        quali_a = st.slider("Quali Position A", 1, 20, 1, key='quali_a')
        standing_a = st.number_input("Championship Points A", 0, 500, 100, key='standing_a')
        home_a = st.toggle("Home Race A", key='home_a')
    
    with col2:
        st.markdown("### Driver B")
        driver_b = st.selectbox("Select Driver B", drivers, key='driver_b')
        team_b = st.selectbox("Select Team B", teams, key='team_b')
        grid_b = st.slider("Grid Position B", 1, 20, 2, key='grid_b')
        quali_b = st.slider("Quali Position B", 1, 20, 2, key='quali_b')
        standing_b = st.number_input("Championship Points B", 0, 500, 100, key='standing_b')
        home_b = st.toggle("Home Race B", key='home_b')
    
    circuit_h2h = st.selectbox("Select Circuit", circuits, key='circuit_h2h')
    rainfall_h2h = st.toggle("🌧️ Wet Race?", key='rainfall_h2h')
    air_temp_h2h = st.slider("Air Temp (°C)", 10, 50, 25, key='air_h2h')
    track_temp_h2h = st.slider("Track Temp (°C)", 15, 70, 35, key='track_h2h')
    
    if st.button("⚔️ Compare", width='stretch'):
        circuit_enc = encoders['circuit'].transform([circuit_h2h])[0]
        
        # Driver A prediction
        driver_a_enc = encoders['driver'].transform([driver_a])[0]
        team_a_enc = encoders['team'].transform([team_a])[0]
        input_a = np.array([[driver_a_enc, team_a_enc, circuit_enc,
                              grid_a, quali_a, int(rainfall_h2h),
                              air_temp_h2h, track_temp_h2h, standing_a, int(home_a)]])
        prob_a = model.predict_proba(scaler.transform(input_a))[0][1]
        
        # Driver B prediction
        driver_b_enc = encoders['driver'].transform([driver_b])[0]
        team_b_enc = encoders['team'].transform([team_b])[0]
        input_b = np.array([[driver_b_enc, team_b_enc, circuit_enc,
                              grid_b, quali_b, int(rainfall_h2h),
                              air_temp_h2h, track_temp_h2h, standing_b, int(home_b)]])
        prob_b = model.predict_proba(scaler.transform(input_b))[0][1]
        
        # Results
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric(f"{driver_a} Win Probability", f"{prob_a*100:.1f}%")
        with col_r2:
            st.metric(f"{driver_b} Win Probability", f"{prob_b*100:.1f}%")
        
        winner = driver_a if prob_a > prob_b else driver_b
        st.success(f"🏆 Model favors **{winner}** in this matchup!")
        
        # Comparison bar chart
        fig7 = go.Figure(data=[
            go.Bar(name=driver_a, x=['Win Probability'], y=[prob_a*100], marker_color='#e8002d'),
            go.Bar(name=driver_b, x=['Win Probability'], y=[prob_b*100], marker_color='#0090d4')
        ])
        fig7.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=350
        )
        st.plotly_chart(fig7, width='stretch')
with tab6:
    st.subheader("🧠 Why Does the Model Predict What It Predicts?")
    
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** reveals how much 
    each feature contributes to the model's predictions.
    Higher value = more influence on the outcome.
    """)
    
    # ── Global SHAP importance chart ──────────────────────────
    st.markdown("### 📊 Overall Feature Importance (SHAP)")
    
    shap_df = pd.DataFrame({
        'Feature': list(shap_data.keys()),
        'SHAP Value': list(shap_data.values())
    }).sort_values('SHAP Value', ascending=True)
    
    fig_shap = px.bar(
        shap_df,
        x='SHAP Value',
        y='Feature',
        orientation='h',
        color='SHAP Value',
        color_continuous_scale='RdYlGn',
        title='Global SHAP Feature Importance'
    )
    fig_shap.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=450
    )
    st.plotly_chart(fig_shap, width='stretch')
    
    # ── Key insights ──────────────────────────────────────────
    st.markdown("### 💡 Key Insights from SHAP Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🏎️ Qualifying is King**
        
        QualiPosition is by far the most influential 
        feature (5.16) — where you qualify determines 
        race outcome more than any other factor.
        This aligns with F1 reality: pole position 
        winners historically win ~40% of races.
        """)
        
        st.info("""
        **📈 Momentum Matters**
        
        Championship StandingPoints (1.78) ranks 2nd — 
        drivers leading the championship tend to 
        perform consistently better, suggesting 
        psychological and team resource advantages.
        """)
    
    with col2:
        st.info("""
        **🏟️ Circuit DNA**
        
        Circuit (1.05) ranks 4th — certain drivers 
        dominate specific tracks consistently. 
        Verstappen at Zandvoort, Hamilton at Silverstone 
        — the model has learned these patterns.
        """)
        
        st.info("""
        **🌧️ Weather is Overrated**
        
        Surprisingly, Rainfall (0.06) has minimal 
        impact in this model — possibly because wet 
        races are rare across 5 seasons of data, 
        limiting the model's ability to learn 
        rain-specific patterns.
        """)
    
    # ── Individual prediction SHAP ─────────────────────────────
    st.markdown("### 🔬 Predict + Explain for a Specific Driver")
    st.markdown("Select inputs below to see SHAP contribution for that prediction:")
    
    col3, col4 = st.columns(2)
    with col3:
        shap_driver = st.selectbox("Driver", drivers, key='shap_driver')
        shap_team = st.selectbox("Team", teams, key='shap_team')
        shap_circuit = st.selectbox("Circuit", circuits, key='shap_circuit')
        shap_grid = st.slider("Grid Position", 1, 20, 1, key='shap_grid')
        shap_quali = st.slider("Quali Position", 1, 20, 1, key='shap_quali')
    with col4:
        shap_points = st.number_input("Championship Points", 
                                       0, 500, 100, key='shap_points')
        shap_rainfall = st.toggle("🌧️ Wet Race?", key='shap_rain')
        shap_air = st.slider("Air Temp (°C)", 10, 50, 25, key='shap_air')
        shap_track = st.slider("Track Temp (°C)", 15, 70, 35, key='shap_track')
        shap_home = st.toggle("🏠 Home Race?", key='shap_home')
    
    if st.button("🧠 Explain This Prediction", key='shap_btn'):
        # Encode inputs
        driver_enc = encoders['driver'].transform([shap_driver])[0]
        team_enc = encoders['team'].transform([shap_team])[0]
        circuit_enc = encoders['circuit'].transform([shap_circuit])[0]
        
        input_data = np.array([[
            driver_enc, team_enc, circuit_enc,
            shap_grid, shap_quali,
            int(shap_rainfall), shap_air, shap_track,
            shap_points, int(shap_home)
        ]])
        
        input_scaled = scaler.transform(input_data)
        
        # Get prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Get SHAP values for this specific prediction
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_scaled)
        
        # shap_vals[1] = SHAP values for class 1 (Win)
        individual_shap = dict(zip(
            ['Driver', 'Team', 'Circuit', 'GridPosition', 
             'QualiPosition', 'Rainfall', 'AirTemp', 
             'TrackTemp', 'StandingPoints', 'HomeRace'],
            shap_vals[0]
        ))
        
        # Show prediction result
        if prediction == 1:
            st.success(f"🏆 {shap_driver} predicted to WIN! "
                      f"({probability*100:.1f}% probability)")
        else:
            st.error(f"❌ {shap_driver} NOT predicted to win "
                    f"({probability*100:.1f}% win probability)")
        
        # Show individual SHAP waterfall
        indiv_df = pd.DataFrame({
            'Feature': list(individual_shap.keys()),
            'SHAP Contribution': list(individual_shap.values())
        }).sort_values('SHAP Contribution', ascending=True)
        
        colors = ['red' if x < 0 else 'green' 
                  for x in indiv_df['SHAP Contribution']]
        
        fig_indiv = px.bar(
            indiv_df,
            x='SHAP Contribution',
            y='Feature',
            orientation='h',
            color='SHAP Contribution',
            color_continuous_scale='RdYlGn',
            title=f'SHAP Explanation — Why {shap_driver} '
                  f'{"wins" if prediction==1 else "doesn\'t win"}'
        )
        fig_indiv.add_vline(x=0, line_dash="dash", 
                            line_color="white", opacity=0.5)
        fig_indiv.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig_indiv, width='stretch')
        
        st.markdown("""
        **How to read this chart:**
        - 🟢 Green bars = features pushing toward a WIN prediction
        - 🔴 Red bars = features pushing against a win prediction
        - Longer bar = stronger influence on this specific prediction
        """)
with tab7:
    st.subheader("🎮 Full Race Grid Simulator")
    st.markdown("""
    Simulate an entire race weekend! The default grid is based 
    on 2023 championship standings. Reorder drivers to simulate 
    different qualifying scenarios.
    """)
    
    # ── Race conditions ────────────────────────────────────────
    st.markdown("### 🌤️ Race Conditions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sim_circuit = st.selectbox("Circuit", circuits, key='sim_circuit')
    with col2:
        sim_rainfall = st.toggle("🌧️ Wet Race?", key='sim_rain')
    with col3:
        sim_air = st.slider("Air Temp (°C)", 10, 50, 25, key='sim_air')
    with col4:
        sim_track = st.slider("Track Temp (°C)", 15, 70, 35, key='sim_track')
    
    st.divider()
    
    # ── Grid builder ───────────────────────────────────────────
    st.markdown("### 🏎️ Qualifying Grid")
    st.markdown("Adjust qualifying positions — drag or change numbers to simulate different grids:")
    
    # Build editable grid
    grid_data = []
    
    col_left, col_right = st.columns(2)
    
    # Split 20 drivers into two columns of 10
    for pos in range(1, 21):
        driver_info = default_grid_2023[pos]
        
        if pos <= 10:
            with col_left:
                col_a, col_b, col_c = st.columns([1, 2, 2])
                with col_a:
                    st.markdown(f"**P{pos}**")
                with col_b:
                    selected_driver = st.selectbox(
                        f"Driver P{pos}",
                        drivers,
                        index=drivers.index(driver_info['driver']) 
                              if driver_info['driver'] in drivers else 0,
                        key=f'sim_driver_{pos}',
                        label_visibility='collapsed'
                    )
                with col_c:
                    selected_team = st.selectbox(
                        f"Team P{pos}",
                        teams,
                        index=teams.index(driver_info['team'])
                              if driver_info['team'] in teams else 0,
                        key=f'sim_team_{pos}',
                        label_visibility='collapsed'
                    )
        else:
            with col_right:
                col_a, col_b, col_c = st.columns([1, 2, 2])
                with col_a:
                    st.markdown(f"**P{pos}**")
                with col_b:
                    selected_driver = st.selectbox(
                        f"Driver P{pos}",
                        drivers,
                        index=drivers.index(driver_info['driver'])
                              if driver_info['driver'] in drivers else 0,
                        key=f'sim_driver_{pos}',
                        label_visibility='collapsed'
                    )
                with col_c:
                    selected_team = st.selectbox(
                        f"Team P{pos}",
                        teams,
                        index=teams.index(driver_info['team'])
                              if driver_info['team'] in teams else 0,
                        key=f'sim_team_{pos}',
                        label_visibility='collapsed'
                    )
        
        grid_data.append({
            'QualiPosition': pos,
            'GridPosition': pos,
            'Driver': selected_driver,
            'Team': selected_team,
            'StandingPoints': default_grid_2023[pos]['points']
        })
    
    st.divider()
    
    # ── Simulate button ────────────────────────────────────────
    if st.button("🚀 Simulate Race!", use_container_width=True):
        
        circuit_enc = encoders['circuit'].transform([sim_circuit])[0]
        
        results = []
        
        for driver_data in grid_data:
            try:
                driver_enc = encoders['driver'].transform(
                    [driver_data['Driver']])[0]
                team_enc = encoders['team'].transform(
                    [driver_data['Team']])[0]
                
                input_data = np.array([[
                    driver_enc, team_enc, circuit_enc,
                    driver_data['GridPosition'],
                    driver_data['QualiPosition'],
                    int(sim_rainfall),
                    sim_air, sim_track,
                    driver_data['StandingPoints'],
                    0  # HomeRace — could enhance later
                ]])
                
                input_scaled = scaler.transform(input_data)
                prob = model.predict_proba(input_scaled)[0][1]
                
                results.append({
                    'Driver': driver_data['Driver'],
                    'Team': driver_data['Team'],
                    'QualiPosition': driver_data['QualiPosition'],
                    'WinProbability': round(prob * 100, 2)
                })
                
            except Exception as e:
                results.append({
                    'Driver': driver_data['Driver'],
                    'Team': driver_data['Team'],
                    'QualiPosition': driver_data['QualiPosition'],
                    'WinProbability': 0.0
                })
        
        # Sort by win probability
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            'WinProbability', ascending=False
        ).reset_index(drop=True)
        
        # ── Predicted winner announcement ──────────────────────
        winner = results_df.iloc[0]
        st.success(f"🏆 Predicted Race Winner: **{winner['Driver']}** "
                  f"({winner['WinProbability']}% win probability)")
        
        # ── Podium ─────────────────────────────────────────────
        st.markdown("### 🥇 Predicted Podium")
        pod1, pod2, pod3 = st.columns(3)
        with pod1:
            st.metric("🥇 1st", results_df.iloc[0]['Driver'],
                     f"{results_df.iloc[0]['WinProbability']}%")
        with pod2:
            st.metric("🥈 2nd", results_df.iloc[1]['Driver'],
                     f"{results_df.iloc[1]['WinProbability']}%")
        with pod3:
            st.metric("🥉 3rd", results_df.iloc[2]['Driver'],
                     f"{results_df.iloc[2]['WinProbability']}%")
        
        # ── Full grid probability chart ────────────────────────
        st.markdown("### 📊 Win Probability — Full Grid")
        
        # Color top 3 differently
        colors = []
        for i in range(len(results_df)):
            if i == 0:
                colors.append('#FFD700')    # Gold for winner
            elif i == 1:
                colors.append('#C0C0C0')    # Silver
            elif i == 2:
                colors.append('#CD7F32')    # Bronze
            else:
                colors.append('#e8002d')    # F1 red for rest
        
        fig_sim = go.Figure(go.Bar(
            x=results_df['WinProbability'],
            y=results_df['Driver'],
            orientation='h',
            marker_color=colors,
            text=[f"{p}%" for p in results_df['WinProbability']],
            textposition='outside'
        ))
        fig_sim.update_layout(
            title=f'Race Win Probability — {sim_circuit}',
            xaxis_title='Win Probability (%)',
            yaxis_title='Driver',
            yaxis=dict(autorange='reversed'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=600
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # ── Full results table ─────────────────────────────────
        st.markdown("### 📋 Full Grid Results")
        results_df.index = results_df.index + 1
        results_df.index.name = 'Predicted Position'
        st.dataframe(
            results_df[['Driver', 'Team', 
                        'QualiPosition', 'WinProbability']],
            use_container_width=True
        )


