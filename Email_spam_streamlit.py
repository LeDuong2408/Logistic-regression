import streamlit as st
import Model_spam_email as model


st.title('Spam Email Detection App')

email_input = st.text_input('Enter an email:', 'Type here...')

if st.button('Predict'):
    prediction = model.email_classification(email_input)

    if prediction == 1:
        st.write('This email is spam.')
    else:
        st.write('This email is not spam.')
