import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Winner Predictor",
    page_icon="🏎️",
    layout="wide"
)

# ── Load model, scaler, encoders, data ───────────────────────
@st.cache_resource
def load_models():
    with open('f1_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('f1_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('f1_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

@st.cache_data
def load_data():
    return pd.read_csv('f1_multiyear.csv')

model, scaler, encoders = load_models()
df = load_data()

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
tab1, tab2, tab3 = st.tabs([
    "🏁 Prediction", 
    "📊 Season Analysis", 
    "🏆 Constructor Stats"
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