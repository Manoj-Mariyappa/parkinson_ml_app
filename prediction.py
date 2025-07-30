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
if selected == 'Clinical Data Test':
    st.title("Parkinson's Voice Test")
    st.write("Enter your voice data values to check for Parkinson's signs")

    col1, col2 = st.columns(2)

    with col1:
        PPE = st.text_input('PPE Value')
        Fo = st.text_input('MDVP:Fo(Hz)')
        Flo = st.text_input('MDVP:Flo(Hz)')
        DDP = st.text_input('Jitter:DDP')
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col2:
        spread1 = st.text_input('Spread1')
        spread2 = st.text_input('Spread2')
        Fhi = st.text_input('MDVP:Fhi(Hz)')
        NHR = st.text_input('NHR')
        APQ5 = st.text_input('Shimmer:APQ5')

    if st.button("\U0001F50D Check Result"):
        if parkinson_model and scaler:
            try:
                user_input = [float(PPE), float(spread1), float(Fo), float(spread2),
                              float(Flo), float(Fhi), float(DDP), float(NHR),
                              float(Jitter_Abs), float(APQ5)]

                input_array = np.array(user_input).reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                prediction = parkinson_model.predict(input_scaled)

                if prediction[0] == 1:
                    st.error("\u26A0\ufe0f **Test shows signs of Parkinson's**")
                    st.write("**Next steps:**")
                    st.write("• Visit a neurologist immediately")
                    st.write("• Don't panic - early detection helps")
                    st.write("• Start regular exercise")
                else:
                    st.success("\u2705 **Test shows no signs of Parkinson's**")
                    st.write("**Keep healthy:**")
                    st.write("• Continue regular exercise")
                    st.write("• Maintain healthy diet")
                    st.write("• Stay socially active")

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
        st.session_state.chat_history = []

    # Get user message using chat_input (auto-submit on Enter)
    user_input = st.chat_input("Ask your question about Parkinson's...")

    if user_input:
        try:
            co = cohere.Client(st.secrets["cohere_api_key"])

            # Add preamble to keep it Parkinson-focused
            response = co.chat(
                message=user_input,
                model="command-r-plus",
                temperature=0.4,
                preamble="You are a helpful medical assistant. Only answer questions related to Parkinson’s disease. "
                         "If a question is not related, respond with: 'Sorry, I can only help with Parkinson’s-related queries.'",
                chat_history=[
                    {"role": "USER", "message": msg["user"]}
                    if msg["role"] == "user" else
                    {"role": "CHATBOT", "message": msg["bot"]}
                    for msg in st.session_state.chat_history
                ],
            )

            # Store messages
            st.session_state.chat_history.append({"role": "user", "user": user_input})
            st.session_state.chat_history.append({"role": "bot", "bot": response.text})

        except Exception as e:
            st.error("⚠️ Error: Check your API key or internet connection.")

    # Display as chat bubbles
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["user"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["bot"])

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
