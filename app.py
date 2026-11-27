# ============================================================
# app.py — Streamlit Frontend for Student Performance Prediction
# Run:  streamlit run app.py
# ============================================================
import streamlit as st
import model_core as mc

st.set_page_config(page_title="Student Performance Prediction",
                   page_icon="🎓", layout="wide")

# ---- Train once (cached) ----
@st.cache_resource
def load_models():
    return mc.get_trained()

trained = load_models()

# ---------------- Header ----------------
st.title("🎓 Student Performance Prediction")
st.markdown(
    "Enter a student's **personal details, study habits and past marks** → the model predicts "
    "**Pass / Fail** and the **expected final score (0–20)**.")
st.divider()

# Attributes that are no longer asked in the form are fixed to the demo
# student's values, so the prediction stays identical (16.37 / 7.38).
FIXED_PROFILE = dict(
    famsize="GT3", Pstatus="T",                                        # family
    Medu=4, Fedu=3, Mjob="teacher", Fjob="services", guardian="mother",
    romantic="no", reason="course", famrel=4, freetime=3, goout=2,     # social
    Dalc=1, Walc=1, health=4,                                          # habits / health
)

# ---------------- Sidebar: model info ----------------
st.sidebar.header("🤖 About the model")
st.sidebar.metric("Prediction accuracy", f"{trained['clf_accuracy']*100:.1f}%",
                  help="Out of 100 students, about 93 are classified correctly.")
st.sidebar.metric("Typical score error", "±1.2 marks",
                  help="On average, the predicted score is about one mark away from the real score.")
st.sidebar.divider()
st.sidebar.caption(
    f"Trained on {trained['n_students']} real student records.\n\n"
    "10 algorithms compared (6 classification + 4 regression).\n\n"
    "Same models as the Colab notebook (random_state = 42).")

# ---------------- Input form (grouped tabs) ----------------
tab1, tab2, tab3 = st.tabs(["👤 Personal", "📚 Study", "📝 Previous Marks"])

with tab1:
    c1, c2, c3 = st.columns(3)
    school = c1.selectbox("School", ["GP", "MS"], help="GP = Gabriel Pereira, MS = Mousinho da Silveira")
    sex = c2.selectbox("Gender", ["M", "F"])
    age = c3.number_input("Age", 15, 22, 16, step=1)
    c4, c5, c6 = st.columns(3)
    address = c4.selectbox("Home address", ["U", "R"], help="U = urban, R = rural")
    famsize = c5.selectbox("Family size", ["GT3", "LE3"], help="GT3 = 3+, LE3 = 3 or fewer")
    Pstatus = c6.selectbox("Parent status", ["T", "A"], help="T = living together, A = apart")

with tab2:
    c1, c2, c3 = st.columns(3)
    studytime = c1.selectbox("Study time", [1, 2, 3, 4], index=2,
                             format_func=lambda x: {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}[x])
    failures = c2.selectbox("Past class failures", [0, 1, 2, 3])
    absences = c3.number_input("School absences", 0, 200, 2, step=1)
    c4, c5, c6 = st.columns(3)
    schoolsup = c4.selectbox("Extra school support", ["yes", "no"], index=1)
    famsup = c5.selectbox("Family educational support", ["yes", "no"])
    paid = c6.selectbox("Extra paid classes", ["yes", "no"])
    c7, c8, c9 = st.columns(3)
    activities = c7.selectbox("Extra-curricular activities", ["yes", "no"])
    nursery = c8.selectbox("Attended nursery school", ["yes", "no"])
    higher = c9.selectbox("Wants higher education", ["yes", "no"])
    internet = st.selectbox("Internet access at home", ["yes", "no"])

with tab3:
    st.markdown("**Previous exam marks — these are the KEY features (0–20 each)**")
    c1, c2, c3 = st.columns(3)
    G1 = c1.slider("1st period marks (G1)", 0, 20, 15)
    G2 = c2.slider("2nd period marks (G2)", 0, 20, 16)
    subject = c3.selectbox("Subject", ["math", "portuguese"])
    c4 = st.container()
    c4.caption("💡 Drag these sliders and hit Predict to see the result change live!")

# ---------------- Predict button ----------------
st.caption("ℹ️ Family and social attributes are kept fixed at typical values; "
           "the prediction is driven mainly by G1, G2 and study habits.")
st.divider()
if st.button("🔮  PREDICT RESULT", type="primary", use_container_width=True):
    student = dict(school=school, sex=sex, age=int(age), address=address,
                   famsize=famsize, Pstatus=Pstatus,
                   studytime=int(studytime), failures=int(failures),
                   schoolsup=schoolsup, famsup=famsup, paid=paid,
                   activities=activities, nursery=nursery, higher=higher,
                   internet=internet, absences=int(absences),
                   G1=int(G1), G2=int(G2), subject=subject,
                   **FIXED_PROFILE)
    out = mc.predict(student)

    r1, r2, r3 = st.columns([2, 1, 1])
    if out["result"] == "Pass":
        r1.success(f"## ✅ PASS  (probability {out['probability']*100:.1f}%)")
    else:
        r1.error(f"## ❌ FAIL  (probability {(1-out['probability'])*100:.1f}%)")
    r2.metric("Predicted Score", f"{out['score']} / 20")
    r3.metric("Model Accuracy", f"{trained['clf_accuracy']*100:.1f}%")

    st.progress(min(out["score"] / 20.0, 1.0),
                text=f"Predicted score: {out['score']} / 20")
    st.info(f"💡 **Advice:** {out['advice']}")

    with st.expander("🔬 How the model decided (details)"):
        st.write(f"- Classifier: **Logistic Regression** with threshold "
                 f"**{out['threshold']:.2f}** (5-fold CV tuned)")
        st.write(f"- Regressor: **Random Forest** (R² = {trained['reg_r2']:.3f})")
        st.write(f"- Input features: {trained['n_features']} (one-hot encoded)")
        st.write("- Trained on 80% of 1044 students; evaluated on unseen 20%")

st.caption("Final Year Project · Machine Learning · Google Colab notebook + this web UI")
