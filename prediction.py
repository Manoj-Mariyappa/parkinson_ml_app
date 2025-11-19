# -*- coding: utf-8 -*-
"""
Created on Sun Jun 8 14:31:36 2025

@author: Manoj M
Guide: K Sharath 
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import cohere

# Load the trained model
try:
    parkinson_model = pickle.load(open('rftrained_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except:
    st.error("Model files not found. Please check the file paths.")
    parkinson_model = None
    scaler = None

#navigation
with st.sidebar:
    selected = option_menu('PARKINSON DISEASE PREDICTION',
                            ['Clinical Data Test',
                            'Self Assessment',
                            'Chat Helper'],  
                           default_index=0,
                           orientation="horizontal",
                          )

# --------------------- Voice Test Page ---------------------
# --------------------- Voice Test Page ---------------------
if selected == 'Clinical Data Test':
    st.title("Parkinson's Voice Test")
    st.write("Enter your voice data values to check for Parkinson's signs")

    # ---------- INITIALIZE SESSION STATE ----------
    if "PPE" not in st.session_state:
        st.session_state.PPE = ""
        st.session_state.Fo = ""
        st.session_state.Flo = ""
        st.session_state.DDP = ""
        st.session_state.Jitter_Abs = ""
        st.session_state.spread1 = ""
        st.session_state.spread2 = ""
        st.session_state.Fhi = ""
        st.session_state.NHR = ""
        st.session_state.APQ5 = ""

    # ---------- FULL CLEAR BUTTON (MUST BE HERE BEFORE INPUTS) ----------
    if st.button("🧹 Clear"):
        for key in ["PPE","Fo","Flo","DDP","Jitter_Abs","spread1","spread2","Fhi","NHR","APQ5"]:
            st.session_state[key] = ""
        st.rerun()

    # ---------- INPUT FIELDS ----------
    col1, col2 = st.columns(2)

    with col1:
        PPE = st.text_input('PPE Value', st.session_state.PPE, key="PPE")
        Fo = st.text_input('MDVP:Fo(Hz)', st.session_state.Fo, key="Fo")
        Flo = st.text_input('MDVP:Flo(Hz)', st.session_state.Flo, key="Flo")
        DDP = st.text_input('Jitter:DDP', st.session_state.DDP, key="DDP")
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', st.session_state.Jitter_Abs, key="Jitter_Abs")

    with col2:
        spread1 = st.text_input('Spread1', st.session_state.spread1, key="spread1")
        spread2 = st.text_input('Spread2', st.session_state.spread2, key="spread2")
        Fhi = st.text_input('MDVP:Fhi(Hz)', st.session_state.Fhi, key="Fhi")
        NHR = st.text_input('NHR', st.session_state.NHR, key="NHR")
        APQ5 = st.text_input('Shimmer:APQ5', st.session_state.APQ5, key="APQ5")

    # ---------- CHECK RESULT ----------
    if st.button("\U0001F50D Check Result"):
        if parkinson_model and scaler:
            try:
                # ---------------- RANGE VALIDATION ----------------
                try:
                    ppe_v = float(PPE)
                    fo_v = float(Fo)
                    flo_v = float(Flo)
                    ddp_v = float(DDP)
                    jitter_abs_v = float(Jitter_Abs)
                    spread1_v = float(spread1)
                    spread2_v = float(spread2)
                    fhi_v = float(Fhi)
                    nhr_v = float(NHR)
                    apq5_v = float(APQ5)
                except:
                    st.error("❌ Please enter valid numbers only.")
                    st.stop()

                valid = True
                err = ""

                if not (0 <= ppe_v <= 0.8):
                    valid = False; err = "❌ PPE must be between 0 and 0.8"
                elif not (60 <= fo_v <= 260):
                    valid = False; err = "❌ MDVP:Fo(Hz) must be between 60 and 260 Hz"
                elif not (60 <= flo_v <= 200):
                    valid = False; err = "❌ MDVP:Flo(Hz) must be between 60 and 200 Hz"
                elif not (0 <= ddp_v <= 0.03):
                    valid = False; err = "❌ Jitter:DDP must be between 0 and 0.03"
                elif not (0 <= jitter_abs_v <= 0.001):
                    valid = False; err = "❌ MDVP:Jitter(Abs) must be between 0 and 0.001"
                elif not (-7 <= spread1_v <= -1):
                    valid = False; err = "❌ Spread1 must be between -7 and -1"
                elif not (0 <= spread2_v <= 0.5):
                    valid = False; err = "❌ Spread2 must be between 0 and 0.5"
                elif not (100 <= fhi_v <= 600):
                    valid = False; err = "❌ MDVP:Fhi(Hz) must be between 100 and 600 Hz"
                elif not (0 <= nhr_v <= 0.6):
                    valid = False; err = "❌ NHR must be between 0 and 0.6"
                elif not (0 <= apq5_v <= 0.05):
                    valid = False; err = "❌ Shimmer:APQ5 must be between 0 and 0.05"

                if not valid:
                    st.error(err)
                    st.stop()
                # ---------------------------------------------------

                user_input = [ppe_v, spread1_v, fo_v, spread2_v,
                              flo_v, fhi_v, ddp_v, nhr_v,
                              jitter_abs_v, apq5_v]

                input_array = np.array(user_input).reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                prediction = parkinson_model.predict(input_scaled)

                if prediction[0] == 1:
                    st.error("\u26A0\ufe0f **Test shows signs of Parkinson's**")
                else:
                    st.success("\u2705 **Test shows no signs of Parkinson's**")

            except:
                st.error("❌ Please enter valid numbers only")
        else:
            st.error("❌ Model not loaded properly")

# --------------------- Self Assessment Page ---------------------
if selected == 'Self Assessment':
    st.title("\U0001F4DD Parkinson's Self Check - Symptom Questions")
    st.write("Answer these questions honestly about your recent experiences")

    if "user_answers" not in st.session_state:
        st.session_state.user_answers = [None] * 7

    questions = [
        {"q": "Do you have hand shaking/tremors?", "options": ["Never", "Sometimes", "Often", "Always"]},
        {"q": "Do you have balance problems?", "options": ["Never", "Sometimes", "Often", "Always"]},
        {"q": "Do you move slower than before?", "options": ["Never", "Sometimes", "Often", "Always"]},
        {"q": "Do you feel body stiffness?", "options": ["Never", "Sometimes", "Often", "Always"]},
        {"q": "Has your voice become softer?", "options": ["No change", "Little soft", "Very soft", "Hard to hear"]},
        {"q": "Has your handwriting become smaller?", "options": ["Same size", "Little smaller", "Much smaller", "Very tiny"]},
        {"q": "Do you feel sad or worried often?", "options": ["Never", "Sometimes", "Often", "Always"]},
    ]

    scores = {"Never": 0, "No change": 0, "Same size": 0, 
              "Sometimes": 1, "Little soft": 1, "Little smaller": 1,
              "Often": 2, "Very soft": 2, "Much smaller": 2,
              "Always": 3, "Hard to hear": 3, "Very tiny": 3}

    st.subheader("Questions")

    for idx, q in enumerate(questions):
        answer = st.radio(f"{idx+1}. {q['q']}", q["options"], index=q["options"].index(st.session_state.user_answers[idx]) if st.session_state.user_answers[idx] else 0, key=f"q_{idx}")
        st.session_state.user_answers[idx] = answer

    if None not in st.session_state.user_answers:
        if st.button("✅ Complete Assessment"):
            total_score = sum([scores[ans] for ans in st.session_state.user_answers])
            risk_percentage = (total_score / 21) * 100

            st.subheader("\U0001F4CA Your Assessment Results")
            st.progress(risk_percentage / 100)

            if risk_percentage < 25:
                st.success(f"🟢 **Low Risk** - {risk_percentage:.0f}%")
                st.write("• Continue healthy habits\n• Monitor any symptom changes")
            elif risk_percentage < 50:
                st.warning(f"🟡 **Medium Risk** - {risk_percentage:.0f}%")
                st.write("• Consult a doctor within 2-4 weeks\n• Start regular physical activity")
            else:
                st.error(f"🔴 **High Risk** - {risk_percentage:.0f}%")
                st.write("• See a neurologist immediately\n• Early treatment is most effective")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Take Test Again"):
                    st.session_state.user_answers = [None] * 7
                    st.rerun()
            with col2:
                st.info("This self-assessment is not a diagnosis. Always consult a doctor if unsure.")
    else:
        st.warning("Please answer all questions to see your result.")


# --------------------- Chat Helper Page ---------------------
if selected == "Chat Helper":
    import cohere
    st.title("💬 Parkinson's Chat Helper")
    st.markdown("Ask Parkinson's-related medical questions.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": "You are a helpful medical assistant. Only answer questions related to Parkinson’s disease."}]

    # Get user message
    user_input = st.chat_input("Ask your question about Parkinson's...")

    if user_input:
        try:
            co = cohere.ClientV2(api_key=st.secrets["cohere_api_key"])

            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Get response from Cohere Chat API
            response = co.chat(
                model="command-r-08-2024",  # active model
                messages=st.session_state.chat_history,
                temperature=0.4
            )

            # Store bot response
            bot_message = response.message.content[0].text
            st.session_state.chat_history.append({"role": "assistant", "content": bot_message})

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    # Display chat messages
    for msg in st.session_state.chat_history[1:]:  
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = [{"role": "system", "content": "You are a helpful medical assistant. Only answer questions related to Parkinson’s disease."}]
        st.rerun()



















