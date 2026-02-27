"""
app.py — Chess Blunder Detector · Streamlit UI
------------------------------------------------
Run:
    streamlit run app.py

Two modes:
    1. PGN Game Analysis  — paste or upload a full PGN game
    2. Single Position    — enter position values manually
"""

import io
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sure src/ is importable when running from project root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.predict import predict_game, predict_position

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chess Blunder Detector",
    page_icon="♟️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(ROOT, "models", "blunder_model.pkl")


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


def colour_row(row):
    """Colour-code rows by blunder probability."""
    p = row["Blunder_Probability"]
    if p >= 0.75:
        return ["background-color: #f8d7da; color: #721c24"] * len(row)   # red
    elif p >= 0.50:
        return ["background-color: #fff3cd; color: #856404"] * len(row)   # amber
    else:
        return ["background-color: #d4edda; color: #155724"] * len(row)   # green


def risk_label(p: float) -> str:
    if p >= 0.75:
        return "🔴 HIGH RISK"
    elif p >= 0.50:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"


def blunder_chart(df: pd.DataFrame) -> go.Figure:
    """Plotly line chart of blunder probability per half-move."""
    fig = go.Figure()

    # Shaded danger zone
    fig.add_hrect(y0=0.5, y1=1.0,
                  fillcolor="rgba(220,53,69,0.08)",
                  line_width=0, annotation_text="Danger zone (≥0.50)")

    # Separate traces for White and Black
    for side, colour in [("White", "#4a90d9"), ("Black", "#e07b39")]:
        mask = df["Side"] == side
        sub  = df[mask]
        fig.add_trace(go.Scatter(
            x=sub.index,
            y=sub["Blunder_Probability"],
            mode="lines+markers",
            name=side,
            line=dict(color=colour, width=2),
            marker=dict(size=5),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Blunder prob: %{y:.1%}<extra></extra>"
            ),
            customdata=sub[["Move_SAN", "Side"]].values,
        ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="Threshold 0.50")

    fig.update_layout(
        title="Per-Move Blunder Probability",
        xaxis_title="Half-move index",
        yaxis_title="Blunder probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/ChessSet.jpg/320px-ChessSet.jpg",
             use_container_width=True)
    st.title("♟️ Chess Blunder Detector")
    st.markdown(
        """
        Predict how likely the **next move** is to be a blunder
        (a move that loses ≥200 centipawns of advantage).

        ---
        **Model:** XGBoost  
        **Target accuracy:** 70–80 %  
        **No engine required** at inference time.
        """
    )
    st.markdown("---")
    if model_exists():
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not found — run `blunder.ipynb` first to train and save it.")

# ---------------------------------------------------------------------------
# Guard: model must exist
# ---------------------------------------------------------------------------
if not model_exists():
    st.warning(
        "⚠️ No trained model found at `models/blunder_model.pkl`.  \n"
        "Please open **blunder.ipynb** and run all cells first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📄 PGN Game Analysis", "🎯 Single Position Check"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PGN Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Analyse a Full Game")
    st.markdown(
        "Paste your PGN below **or** upload a `.pgn` file.  \n"
        "Export your games from [Chess.com](https://www.chess.com/games) "
        "or [Lichess](https://lichess.org/games/search)."
    )

    col_paste, col_upload = st.columns([3, 1])

    with col_paste:
        pgn_text = st.text_area(
            "Paste PGN here",
            height=220,
            placeholder=(
                '[White "YourName"]\n'
                '[Black "Opponent"]\n'
                '[WhiteElo "1500"]\n'
                '[BlackElo "1400"]\n'
                '[TimeControl "600+0"]\n\n'
                "1. e4 e5 2. Nf3 Nc6 ..."
            ),
        )

    with col_upload:
        st.markdown("**Or upload a file:**")
        uploaded = st.file_uploader("Upload .pgn", type=["pgn"])
        if uploaded:
            pgn_text = uploaded.read().decode("utf-8", errors="ignore")
            st.success(f"Loaded: {uploaded.name}")

    analyse_btn = st.button("🔍 Analyse Game", type="primary", use_container_width=True)

    if analyse_btn:
        if not pgn_text or pgn_text.strip() == "":
            st.warning("Please paste or upload a PGN first.")
        else:
            with st.spinner("Analysing moves …"):
                try:
                    # Write PGN to a temp file so predict_game can read it
                    tmp_path = os.path.join(ROOT, "_tmp_game.pgn")
                    with open(tmp_path, "w", encoding="utf-8") as fh:
                        fh.write(pgn_text)

                    results = predict_game(tmp_path, MODEL_PATH)

                    os.remove(tmp_path)

                    # ---- Summary metrics ----
                    n_total    = len(results)
                    n_blunders = int((results["Blunder_Probability"] >= 0.5).sum())
                    avg_prob   = results["Blunder_Probability"].mean()
                    worst_move = results.loc[results["Blunder_Probability"].idxmax()]

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Moves",      n_total)
                    m2.metric("Suspected Blunders (≥0.50)", n_blunders)
                    m3.metric("Avg Blunder Prob", f"{avg_prob:.1%}")
                    m4.metric("Riskiest Move",
                              f"Move {worst_move['Move_Number']} — {worst_move['Move_SAN']}",
                              f"{worst_move['Blunder_Probability']:.1%} ({worst_move['Side']})")

                    st.markdown("---")

                    # ---- Chart ----
                    st.plotly_chart(blunder_chart(results), use_container_width=True)

                    # ---- Table ----
                    st.subheader("Move-by-Move Breakdown")
                    styled = (
                        results
                        .style
                        .apply(colour_row, axis=1)
                        .format({"Blunder_Probability": "{:.1%}"})
                    )
                    st.dataframe(styled, use_container_width=True, height=420)

                    # ---- Download ----
                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download results as CSV",
                        csv, "blunder_analysis.csv", "text/csv",
                    )

                except Exception as e:
                    st.error(f"Error analysing game: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Single position
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Check a Single Position")
    st.markdown(
        "Don't have a PGN? Enter the details of your position manually "
        "and get an instant blunder risk estimate."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Player Info")
        white_elo = st.slider("Your rating (White ELO)",  500, 3000, 1500, step=10)
        black_elo = st.slider("Opponent rating (Black ELO)", 500, 3000, 1400, step=10)
        base_time = st.selectbox(
            "Time control",
            options=[60, 180, 300, 600, 900, 1800, 3600],
            format_func=lambda x: {
                60: "1 min (Bullet)",
                180: "3 min (Bullet)",
                300: "5 min (Blitz)",
                600: "10 min (Blitz)",
                900: "15 min (Rapid)",
                1800: "30 min (Rapid)",
                3600: "60 min (Classical)",
            }.get(x, f"{x}s"),
            index=3,
        )

    with c2:
        st.subheader("Board State")
        total_pieces  = st.slider("Total pieces on board", 2, 32, 28)
        material_diff = st.slider("Material balance (centipawns, + = White ahead)",
                                  -900, 900, 0, step=50)
        in_check      = st.radio("Is the side to move in check?",
                                 options=[0, 1],
                                 format_func=lambda x: "Yes ✓" if x else "No",
                                 horizontal=True)

    predict_btn = st.button("⚡ Predict Blunder Risk", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Calculating …"):
            prob = predict_position(
                sample_dict={
                    "white_elo":     white_elo,
                    "black_elo":     black_elo,
                    "depth":         20,
                    "material_diff": material_diff,
                    "total_pieces":  total_pieces,
                    "in_check":      int(in_check),
                    "rating_diff":   white_elo - black_elo,
                    "avg_rating":    (white_elo + black_elo) / 2,
                    "base_time":     float(base_time),
                },
                model_path=MODEL_PATH,
            )

        # Large probability display
        label = risk_label(prob)
        colour = "#721c24" if prob >= 0.75 else ("#856404" if prob >= 0.5 else "#155724")
        bg     = "#f8d7da"  if prob >= 0.75 else ("#fff3cd" if prob >= 0.5 else "#d4edda")

        st.markdown(
            f"""
            <div style="
                background-color:{bg};
                border-radius:12px;
                padding:28px 36px;
                margin-top:20px;
                text-align:center;
            ">
                <div style="font-size:2.4rem; font-weight:700; color:{colour};">{label}</div>
                <div style="font-size:3.6rem; font-weight:900; color:{colour}; margin:10px 0;">
                    {prob:.1%}
                </div>
                <div style="font-size:1rem; color:{colour};">
                    Estimated probability that the next move is a blunder (≥200 cp drop)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": colour},
                "steps": [
                    {"range": [0,  50], "color": "#d4edda"},
                    {"range": [50, 75], "color": "#fff3cd"},
                    {"range": [75, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            title={"text": "Blunder Risk"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=20, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Interpretation
        with st.expander("ℹ️ How to interpret this"):
            st.markdown(
                f"""
                | Range | Meaning |
                |---|---|
                | 🟢 0–50% | Move is likely fine — position is manageable |
                | 🟡 50–75% | Caution — significant blunder risk, double-check your move |
                | 🔴 75–100% | High risk — very likely to blunder here, think carefully |

                **Your inputs:**
                - ELO: {white_elo} (you) vs {black_elo} (opponent)
                - Time control: {base_time}s base
                - Pieces on board: {total_pieces} → {"Endgame" if total_pieces < 12 else "Middlegame" if total_pieces < 24 else "Opening/Early middlegame"}
                - In check: {"Yes" if in_check else "No"}
                - Material: {"Even" if material_diff == 0 else f"+{material_diff} cp (White)" if material_diff > 0 else f"{material_diff} cp (Black ahead)"}
                """
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Chess Blunder Detector · XGBoost classifier · "
    "Target accuracy 70–80% · No engine required at inference time"
)
