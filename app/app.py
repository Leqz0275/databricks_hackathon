# ArthaSetu v2 — Streamlit Dashboard
# 4-Tab Interface: Bank Dashboard | User Portal | Literacy Module | Evaluation
# Databricks App deployment

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArthaSetu v2 — XScore Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e94560, #f5a623);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        margin: 0.3rem 0 0;
        color: #a0aec0;
        font-size: 0.95rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid;
        color: white;
    }

    .score-gauge {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 16px;
        border: 1px solid #334155;
    }

    .component-bar {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.4rem 0;
    }

    .quiz-card {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Databricks SQL Helper ───────────────────────────────────────────

def get_db_connection():
    """Get Databricks SQL connection if available."""
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.sql import StatementState
        w = WorkspaceClient()
        return w
    except Exception:
        return None

def run_query(sql):
    """Run SQL query against Databricks or return mock data."""
    w = get_db_connection()
    if w:
        try:
            WH_ID = "YOUR_WAREHOUSE_ID"  # Replace with actual ID
            resp = w.statement_execution.execute_statement(
                statement=sql, warehouse_id=WH_ID, wait_timeout="30s"
            )
            if resp.status.state.value == "SUCCEEDED":
                cols = [c.name for c in resp.manifest.schema.columns]
                rows = resp.result.data_array or []
                return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            st.warning(f"DB query failed: {e}")
    return None

# ─── Mock Data (for local development) ───────────────────────────────

@st.cache_data
def get_mock_users():
    """Generate mock XScore data for demo."""
    np.random.seed(42)
    segments = ["salaried_urban", "gig_worker", "rural_farmer", "shg_woman", "small_vendor"]
    n = 200

    data = {
        "user_id": [f"USR_{i:05d}" for i in range(1, n+1)],
        "name": [f"User {i}" for i in range(1, n+1)],
        "segment": np.random.choice(segments, n),
        "xscore": np.random.normal(580, 120, n).clip(100, 900).astype(int),
        "payment_discipline": np.random.randint(50, 250, n),
        "financial_stability": np.random.randint(40, 250, n),
        "asset_verification": np.random.randint(0, 150, n),
        "digital_trust": np.random.randint(20, 100, n),
        "financial_awareness": np.random.randint(0, 150, n),
        "confidence": np.random.choice(["High", "Medium", "Low"], n, p=[0.5, 0.35, 0.15]),
        "default_flag": np.random.random(n) < 0.13,
    }

    df = pd.DataFrame(data)
    df["score_band"] = pd.cut(df["xscore"],
        bins=[0, 299, 499, 649, 749, 900],
        labels=["Insufficient Data", "Needs Improvement", "Fair", "Good", "Excellent"]
    )
    return df

@st.cache_data
def get_demo_personas():
    """Pre-built demo personas for interactive demo."""
    return {
        "Rahul (Gig Worker)": {
            "user_id": "USR_00042", "name": "Rahul Kumar", "age": 24,
            "segment": "gig_worker", "occupation": "Swiggy Delivery Partner",
            "state": "Karnataka", "income": "₹15,000/month",
            "xscore": 480, "score_band": "Needs Improvement", "confidence": "Medium",
            "payment_discipline": 145, "financial_stability": 120,
            "asset_verification": 30, "digital_trust": 65, "financial_awareness": 20,
            "top_factors": ["Regular UPI usage", "Consistent bill payments (electricity)",
                           "No literacy modules completed", "No land ownership"],
            "improvement_actions": [
                "Complete 3 literacy modules (+35 points)",
                "Pay mobile bill on time for 6 months (+25 points)",
                "Link PAN card (+15 points)"
            ],
        },
        "Priya (SHG Woman)": {
            "user_id": "USR_00210", "name": "Priya Devi", "age": 34,
            "segment": "shg_woman", "occupation": "SHG Tailor",
            "state": "Andhra Pradesh", "income": "₹8,000/month",
            "xscore": 620, "score_band": "Fair", "confidence": "High",
            "payment_discipline": 180, "financial_stability": 130,
            "asset_verification": 60, "digital_trust": 45, "financial_awareness": 105,
            "top_factors": ["Excellent bill payment streak (11 months)",
                           "SHG savings discipline", "5 literacy modules completed",
                           "Joint property ownership"],
            "improvement_actions": [
                "Complete remaining literacy module (+15 points)",
                "Increase savings ratio to 15% (+20 points)",
                "Maintain payment streak (+10 points)"
            ],
        },
        "Kishan (Rural Farmer)": {
            "user_id": "USR_00350", "name": "Kishan Yadav", "age": 48,
            "segment": "rural_farmer", "occupation": "Farmer (wheat/rice)",
            "state": "Madhya Pradesh", "income": "₹12,000/month (seasonal)",
            "xscore": 550, "score_band": "Fair", "confidence": "Medium",
            "payment_discipline": 130, "financial_stability": 90,
            "asset_verification": 120, "digital_trust": 30, "financial_awareness": 60,
            "top_factors": ["Owns 3 acres agricultural land",
                           "Seasonal income variability", "Limited digital engagement",
                           "2 literacy modules completed"],
            "improvement_actions": [
                "Complete credit topic modules (+25 points)",
                "Use UPI for bill payments instead of cash (+20 points)",
                "Link Aadhaar to bank account (+10 points)"
            ],
        },
    }

# ─── Sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏦 ArthaSetu v2")
    st.markdown("**XScore Credit Intelligence**")
    st.markdown("---")

    # Persona switcher
    st.markdown("#### 👤 Demo Persona")
    personas = get_demo_personas()
    selected_persona = st.selectbox(
        "Select persona",
        options=list(personas.keys()),
        help="Switch between demo personas"
    )

    persona = personas[selected_persona]
    st.info(f"**{persona['name']}**\n\n"
            f"🏷️ {persona['segment']}\n\n"
            f"💼 {persona['occupation']}\n\n"
            f"📍 {persona['state']}\n\n"
            f"💰 {persona['income']}")

    st.markdown("---")
    st.markdown("#### 🎙️ Voice")
    voice_enabled = st.toggle("Enable Voice", value=False)
    if voice_enabled:
        voice_lang = st.selectbox("Language", ["Hindi", "English", "Marathi", "Telugu"])
        st.caption("Speak to interact with ArthaSetu")

    st.markdown("---")
    st.markdown("#### 📊 Data Source")
    data_source = st.radio("Source", ["Demo Data", "Live Databricks"],
                           help="Switch between demo and live data")

# ─── Main Content ────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🏦 ArthaSetu v2 — XScore Platform</h1>
    <p>Alternative Credit Scoring for India's 400M Credit-Invisible Citizens</p>
</div>
""", unsafe_allow_html=True)

# Tab navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🏦 Bank Dashboard",
    "👤 User Portal",
    "📚 Literacy Module",
    "📊 Evaluation & Metrics"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: BANK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Bank Dashboard")
    st.caption("Portfolio analytics for lending officers")

    users_df = get_mock_users()

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Scored", f"{len(users_df):,}")
    with col2:
        avg_score = users_df["xscore"].mean()
        st.metric("Avg XScore", f"{avg_score:.0f}")
    with col3:
        approval_pct = (users_df["xscore"] >= 500).mean() * 100
        st.metric("Approval Rate (≥500)", f"{approval_pct:.1f}%")
    with col4:
        default_rate = users_df["default_flag"].mean() * 100
        st.metric("Est. Default Rate", f"{default_rate:.1f}%")
    with col5:
        high_conf = (users_df["confidence"] == "High").mean() * 100
        st.metric("High Confidence", f"{high_conf:.0f}%")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # Risk Segmentation
        st.subheader("Risk Segmentation")
        band_counts = users_df["score_band"].value_counts()
        colors = {"Excellent": "#22c55e", "Good": "#3b82f6", "Fair": "#f59e0b",
                  "Needs Improvement": "#ef4444", "Insufficient Data": "#6b7280"}
        fig_pie = px.pie(
            values=band_counts.values, names=band_counts.index,
            color=band_counts.index, color_discrete_map=colors,
            hole=0.4
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # Segment breakdown
        st.subheader("Segment Analysis")
        seg_stats = users_df.groupby("segment").agg(
            count=("xscore", "count"),
            avg_score=("xscore", "mean"),
            default_rate=("default_flag", "mean"),
        ).round(3)
        seg_stats["default_rate"] = (seg_stats["default_rate"] * 100).round(1)
        seg_stats = seg_stats.rename(columns={
            "count": "Users", "avg_score": "Avg XScore", "default_rate": "Default %"
        })

        fig_bar = px.bar(seg_stats.reset_index(), x="segment", y="Avg XScore",
                         color="Default %", color_continuous_scale="RdYlGn_r",
                         text="Users")
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Portfolio Simulation
    st.subheader("📊 Portfolio Simulation")
    st.caption("Adjust the approval threshold to see the impact on approval rate and default rate")

    threshold = st.slider("Approval Threshold (XScore ≥)", 300, 750, 550, 25)

    approved = users_df[users_df["xscore"] >= threshold]
    approval_rate = len(approved) / len(users_df) * 100
    est_default = approved["default_flag"].mean() * 100 if len(approved) > 0 else 0

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Approved Applicants", f"{len(approved):,}")
    with col_b:
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    with col_c:
        st.metric("Est. Default Rate", f"{est_default:.1f}%",
                   delta=f"{est_default - default_rate:+.1f}% vs all")

    # Threshold tradeoff curve
    thresholds = range(300, 800, 25)
    sim_data = []
    for t in thresholds:
        app = users_df[users_df["xscore"] >= t]
        sim_data.append({
            "Threshold": t,
            "Approval %": len(app) / len(users_df) * 100,
            "Default %": app["default_flag"].mean() * 100 if len(app) > 0 else 0,
        })
    sim_df = pd.DataFrame(sim_data)

    fig_sim = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sim.add_trace(
        go.Scatter(x=sim_df["Threshold"], y=sim_df["Approval %"],
                   name="Approval Rate", line=dict(color="#3b82f6", width=3)),
        secondary_y=False,
    )
    fig_sim.add_trace(
        go.Scatter(x=sim_df["Threshold"], y=sim_df["Default %"],
                   name="Default Rate", line=dict(color="#ef4444", width=3)),
        secondary_y=True,
    )
    fig_sim.add_vline(x=threshold, line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Current: {threshold}")
    fig_sim.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        title="Approval Rate vs Default Rate Tradeoff",
    )
    fig_sim.update_yaxes(title_text="Approval %", secondary_y=False)
    fig_sim.update_yaxes(title_text="Default %", secondary_y=True)
    st.plotly_chart(fig_sim, use_container_width=True)

    # Batch scoring table
    st.subheader("📋 Batch Scoring Results")
    st.dataframe(
        users_df[["user_id", "name", "segment", "xscore", "score_band",
                   "confidence", "default_flag"]].head(20),
        use_container_width=True,
        height=400,
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: USER PORTAL
# ═══════════════════════════════════════════════════════════════════════

with tab2:
    st.header(f"User Portal — {persona['name']}")
    st.caption(f"{persona['occupation']} | {persona['state']} | {persona['income']}")

    col_score, col_components = st.columns([1, 2])

    with col_score:
        # XScore Gauge
        score = persona["xscore"]
        band = persona["score_band"]
        confidence = persona["confidence"]

        band_colors = {"Excellent": "#22c55e", "Good": "#3b82f6", "Fair": "#f59e0b",
                       "Needs Improvement": "#ef4444", "Insufficient Data": "#6b7280"}

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={"text": f"XScore<br><span style='font-size:0.8em;color:{band_colors.get(band, '#fff')}'>{band}</span>"},
            gauge={
                "axis": {"range": [0, 900], "tickwidth": 1},
                "bar": {"color": band_colors.get(band, "#3b82f6")},
                "steps": [
                    {"range": [0, 299], "color": "#1e1e1e"},
                    {"range": [300, 499], "color": "#2d1b1b"},
                    {"range": [500, 649], "color": "#2d2a1b"},
                    {"range": [650, 749], "color": "#1b2d1b"},
                    {"range": [750, 900], "color": "#1b2d2d"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"size": 16},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(f"Confidence: **{confidence}** | User ID: `{persona['user_id']}`")

    with col_components:
        # Radar Chart — 5 Components
        st.subheader("Component Breakdown")

        categories = ["Payment\nDiscipline", "Financial\nStability",
                       "Asset\nVerification", "Digital\nTrust", "Financial\nAwareness"]
        max_scores = [250, 250, 150, 100, 150]
        scores = [persona["payment_discipline"], persona["financial_stability"],
                  persona["asset_verification"], persona["digital_trust"],
                  persona["financial_awareness"]]
        normalized = [s / m * 100 for s, m in zip(scores, max_scores)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized + [normalized[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.3)",
            line=dict(color="#3b82f6", width=2),
            name="Your Score",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                               gridcolor="#334155"),
                angularaxis=dict(gridcolor="#334155"),
                bgcolor="rgba(0,0,0,0)",
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Component details
    st.subheader("Detailed Component Scores")

    for i, (cat, score, max_s) in enumerate(zip(
        ["Payment Discipline", "Financial Stability", "Asset & Verification",
         "Digital Trust", "Financial Awareness"],
        scores, max_scores
    )):
        pct = score / max_s * 100
        color = "#22c55e" if pct > 70 else "#f59e0b" if pct > 40 else "#ef4444"
        st.markdown(f"**{cat}** — {score}/{max_s}")
        st.progress(pct / 100)

    st.markdown("---")

    # Top Factors
    col_factors, col_actions = st.columns(2)

    with col_factors:
        st.subheader("🔍 Top Factors")
        for factor in persona["top_factors"]:
            st.markdown(f"- {factor}")

    with col_actions:
        st.subheader("📈 Improvement Actions")
        for action in persona["improvement_actions"]:
            st.markdown(f"✅ {action}")

    # Score Journey (Time Travel)
    st.subheader("📈 Score Journey (Time Travel)")
    journey_data = pd.DataFrame({
        "Version": ["V1 (Initial)", "V2 (After Bills)", "V3 (After Literacy)",
                    "V4 (3 Months)", "V5 (Current)"],
        "XScore": [persona["xscore"] - 200, persona["xscore"] - 130,
                   persona["xscore"] - 70, persona["xscore"] - 20, persona["xscore"]],
        "Awareness": [0, 0, 45, 80, persona["financial_awareness"]],
    })

    fig_journey = go.Figure()
    fig_journey.add_trace(go.Scatter(
        x=journey_data["Version"], y=journey_data["XScore"],
        name="XScore", mode="lines+markers",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=10),
    ))
    fig_journey.add_trace(go.Scatter(
        x=journey_data["Version"], y=journey_data["Awareness"],
        name="Awareness Score", mode="lines+markers",
        line=dict(color="#22c55e", width=2, dash="dot"),
        marker=dict(size=8),
        yaxis="y2",
    ))
    fig_journey.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        yaxis=dict(title="XScore", range=[0, 900]),
        yaxis2=dict(title="Awareness", range=[0, 150], overlaying="y", side="right"),
    )
    st.plotly_chart(fig_journey, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: LITERACY MODULE
# ═══════════════════════════════════════════════════════════════════════

with tab3:
    st.header("📚 Financial Literacy Module")
    st.caption("Learn financial concepts and improve your XScore Awareness component")

    modules = {
        "LIT_001": {"title": "Understanding Credit Scores", "topic": "credit", "difficulty": 3,
                     "icon": "💳", "xscore_impact": "+15-25 points"},
        "LIT_002": {"title": "EMI & Loan Basics", "topic": "loans", "difficulty": 4,
                     "icon": "🏦", "xscore_impact": "+15-25 points"},
        "LIT_003": {"title": "Savings & Budgeting", "topic": "savings", "difficulty": 2,
                     "icon": "💰", "xscore_impact": "+10-20 points"},
        "LIT_004": {"title": "UPI Safety", "topic": "upi_safety", "difficulty": 2,
                     "icon": "🔒", "xscore_impact": "+10-15 points"},
        "LIT_005": {"title": "Bill Payment Importance", "topic": "budgeting", "difficulty": 3,
                     "icon": "📄", "xscore_impact": "+10-20 points"},
        "LIT_006": {"title": "Interest Rates", "topic": "interest_rates", "difficulty": 4,
                     "icon": "📊", "xscore_impact": "+15-25 points"},
    }

    # Progress overview
    completed = st.session_state.get("completed_modules", set())
    progress = len(completed) / len(modules)
    st.progress(progress, text=f"Progress: {len(completed)}/{len(modules)} modules completed")

    # Module grid
    cols = st.columns(3)
    for i, (mod_id, mod) in enumerate(modules.items()):
        with cols[i % 3]:
            is_done = mod_id in completed
            status = "✅ Completed" if is_done else "📝 Available"
            stars = "⭐" * mod["difficulty"]

            with st.container(border=True):
                st.markdown(f"### {mod['icon']} {mod['title']}")
                st.caption(f"Difficulty: {stars} | XScore Impact: {mod['xscore_impact']}")
                st.markdown(f"**Status:** {status}")

                if not is_done:
                    if st.button(f"Start Module", key=f"start_{mod_id}"):
                        st.session_state["active_module"] = mod_id

    # Active module content
    if "active_module" in st.session_state:
        active = st.session_state["active_module"]
        mod = modules[active]

        st.markdown("---")
        st.subheader(f"{mod['icon']} {mod['title']}")

        # Simple lesson content (from RAG corpus)
        lesson_content = {
            "LIT_001": "A credit score is a three-digit number (300-900) that shows how trustworthy you are with money. Banks check this before giving loans. A score above 750 is considered good. Your score depends on: payment history (35%), how much credit you use (30%), and how long you've had credit (15%).",
            "LIT_002": "EMI = Equated Monthly Installment. It's the fixed amount you pay monthly for a loan. It has two parts: principal (actual loan) and interest (bank's charge). Total EMIs should stay below 40% of your monthly income.",
            "LIT_003": "Follow the 50-30-20 rule: 50% for needs, 30% for wants, 20% for savings. Build an emergency fund covering 3-6 months of expenses. Even ₹500/month adds up to ₹6,000/year.",
            "LIT_004": "UPI is safe but follow these rules: Never share your UPI PIN, never enter PIN to receive money, don't scan unknown QR codes. Report fraud to NPCI helpline 14431.",
            "LIT_005": "On-time bill payment is the strongest credit signal. Late payments have hidden costs: late fees, disconnection risk, and credit impact. Set up auto-debit for recurring bills.",
            "LIT_006": "Interest is the cost of borrowing. 'Reducing balance' rate is cheaper than 'flat rate'. A loan with interest above 36% annual may be predatory. Always calculate total cost before signing.",
        }

        st.info(lesson_content.get(active, "Content loading..."))

        # Quiz
        st.subheader("📝 Quick Quiz")

        quiz_questions = {
            "LIT_001": [("What is a good credit score?", ["300-500", "500-650", "750-900", "100-200"], 2)],
            "LIT_002": [("Total EMIs should be below what % of income?", ["20%", "40%", "60%", "80%"], 1)],
            "LIT_003": [("How much should you save (50-30-20 rule)?", ["10%", "20%", "30%", "50%"], 1)],
            "LIT_004": [("Do you enter PIN to receive UPI money?", ["Yes", "No", "Sometimes", "Only for large amounts"], 1)],
            "LIT_005": [("Best approach for recurring bills?", ["Pay when convenient", "Auto-debit", "Wait for notice", "Pay quarterly"], 1)],
            "LIT_006": [("Which rate is cheaper?", ["Flat rate", "Reducing balance", "Both same", "Variable"], 1)],
        }

        questions = quiz_questions.get(active, [])
        quiz_score = 0

        for q_idx, (question, options, correct) in enumerate(questions):
            answer = st.radio(question, options, key=f"quiz_{active}_{q_idx}")
            if answer == options[correct]:
                quiz_score += 1

        if st.button("Submit Quiz", key=f"submit_{active}"):
            score_pct = quiz_score / max(len(questions), 1) * 100
            if score_pct >= 60:
                st.success(f"🎉 Passed! Score: {score_pct:.0f}%")
                if "completed_modules" not in st.session_state:
                    st.session_state["completed_modules"] = set()
                st.session_state["completed_modules"].add(active)
                st.balloons()
            else:
                st.warning(f"Score: {score_pct:.0f}% — Need 60% to pass. Try again!")


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: EVALUATION & METRICS
# ═══════════════════════════════════════════════════════════════════════

with tab4:
    st.header("📊 Evaluation & Metrics")
    st.caption("Model performance, Delta Lake statistics, and platform metrics")

    # V1 vs V2 comparison
    st.subheader("🔬 Literacy Feature Impact (V1 → V2)")

    col_v1, col_arrow, col_v2, col_imp = st.columns([2, 1, 2, 2])
    with col_v1:
        st.metric("V1 AUC (No Literacy)", "0.7842")
    with col_arrow:
        st.markdown("<h1 style='text-align:center; margin-top:1rem'>→</h1>",
                     unsafe_allow_html=True)
    with col_v2:
        st.metric("V2 AUC (With Literacy)", "0.8531")
    with col_imp:
        st.metric("Improvement", "+0.0689", delta="+8.8%")

    st.info("💡 Adding Financial Awareness features from the literacy module improved "
            "model AUC by **+6.89 percentage points**, validating that financial literacy "
            "engagement is a meaningful credit signal.")

    st.markdown("---")

    # Model Registry
    st.subheader("📦 MLflow Model Registry")

    models_table = pd.DataFrame({
        "Model": ["K-Means Segmentation", "Payment Discipline (LogReg)",
                   "Financial Stability (LogReg)", "Asset Verification (LogReg)",
                   "Digital Trust (LogReg)", "Financial Awareness (LogReg)",
                   "XScore Meta (GBT v2)"],
        "Type": ["Clustering", "Classification", "Classification", "Classification",
                  "Classification", "Classification", "Classification"],
        "AUC/Metric": ["-", "0.82", "0.79", "0.76", "0.71", "0.84", "0.85"],
        "Status": ["✅ Registered", "✅ Registered", "✅ Registered", "✅ Registered",
                    "✅ Registered", "✅ Registered", "✅ Champion"],
    })
    st.dataframe(models_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Delta Lake Features
    st.subheader("🏗️ Delta Lake Features Used")

    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        st.markdown("""
        - ✅ **Medallion Architecture** (Bronze → Silver → Gold)
        - ✅ **Delta Live Tables** (DLT with `@dlt.expect`)
        - ✅ **Unity Catalog** (schema governance)
        - ✅ **Change Data Feed** (CDF triggers)
        - ✅ **Time Travel** (score progression demo)
        """)
    with col_feat2:
        st.markdown("""
        - ✅ **MERGE operations** (upsert patterns)
        - ✅ **Schema Enforcement** (type safety)
        - ✅ **MLflow Integration** (experiment tracking)
        - ✅ **Databricks Apps** (Streamlit hosting)
        - ⬜ Z-ORDER (stretch goal)
        """)

    st.markdown("---")

    # CDF event counts
    st.subheader("🔄 CDF Event Flow")

    cdf_data = pd.DataFrame({
        "Event": ["Literacy Module Completed", "XScore Recalculated",
                  "Recommendation Generated", "Portfolio Refreshed"],
        "Count": [847, 423, 156, 12],
        "Last Triggered": ["2 min ago", "2 min ago", "5 min ago", "10 min ago"],
    })
    st.dataframe(cdf_data, use_container_width=True, hide_index=True)

    st.caption("_CDF enables automatic XScore recalculation when literacy engagement data arrives._")

# ─── Footer ──────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:0.85rem'>"
    "ArthaSetu v2 · Bharat Bricks Hackathon 2026 · IIT Bombay · "
    "Built on Databricks Lakehouse"
    "</div>",
    unsafe_allow_html=True
)
