import streamlit as st
from streamlit_option_menu import option_menu
import openai

# Sidebar for navigation and API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
openai.api_key = api_key

with st.sidebar:
    page = option_menu(
        "Dashboard",
        ["Home", "About Me", "AWS Study Buddy"],
        icons=['house', 'info-circle',  'file-text'],
        menu_icon="list",
        default_index=0,
    )

if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to use the application.")

else:
    if page == "Home":
        st.title("AWS Examination Study Buddy")
        st.write("Hi! I am your AWS Examination Study Buddy! This platform allows you to chat with a Study Buddy Chat Bot to help you gain a better understanding of AWS concepts and prepare you for your upcoming examinations!")

        st.write("## How It Works")
        st.write("1. **Ask me anything AWS-related:** Send me a question that you'd like to be answered. I will answer in a concise manner, limiting my answers to only 400 characters so you will not be overwhelmed with technical jargons! I know, its kinda boring to read lengthy explanations!")
        st.write("2. **Provide follow-up questions:** The tool can be used to converse about any AWS-related topic so you can dive deeper into topics that you are really interested to know.")
        st.write("3. **Let's stick to the topic!:** Since I am tasked to be your AWS Study Buddy, please refrain from asking me questions not related to AWS, oki?")

        st.write("## Ideal Users")
        st.write("This tool is perfect for:")
        st.write("- AWS Solutions Architects and Professionals who would like to understand AWS services better and faster.")
        st.write("- Students and professionals who would like to have a digital study buddy while preparing for their AWS Certification Exams.")
        st.write("- IT Managers who would like to have a high-level overview of AWS services but do not have the pleasure of time to go over long documentations.")

        st.write("Start using the AWS Examination Study Buddy today to boost your AWS knowledge and ace your certification exams!")

    elif page == "About Me":
        st.header("About Me")
        st.markdown("""
        Hi! I'm Erwin Caluag! I am a Software Engineer / Web Developer and an AWS Solutions Architect Associate. Currently, I am venturing into the world of Artificial Intelligence.
                    
        This project is one of the projects I am building to try and apply the learnings I have acquired from the AI First Bootcamp from AI Republic. 
                    
        Any feedback would be greatly appreciated!
        """)

    elif page == "AWS Study Buddy":
        System_Prompt = """
Role: Act as an AWS Solutions Architect expert, guiding me through AWS concepts.

Intent: Help me understand AWS topics in depth by providing clear answers to questions I ask, which will prepare me for the AWS Certified Solutions Architect Associate exam.

Context: I am studying for the AWS Certified Solutions Architect Associate exam and want to deepen my understanding of key AWS services, architecture patterns, and best practices.

Constraints: Please keep explanations concise but thorough, emphasizing practical applications and highlighting any best practices, common pitfalls, or essential configurations. Do not answer anything that is not AWS related. Clearly remind the user that you are only tasked to answer anything AWS related. Limit your answer to only 400 characters.

Example 1: For instance, if I ask about Amazon S3 storage classes, explain the various classes, when to use each, and provide examples of common scenarios.

Example 2: If I ask about Amazon EC2 instance types, outline the key categories (like general-purpose, compute-optimized, and memory-optimized), and give examples of specific use cases for each.

Example 3: If I ask about setting up VPC peering, explain the steps involved, any key configuration considerations, and mention common issues or limitations.
"""

        def initialize_conversation(prompt):
            if 'message' not in st.session_state:
                st.session_state.message = []
                st.session_state.message.append({"role": "system", "content": System_Prompt})

        initialize_conversation(System_Prompt)

        for messages in st.session_state.message:
            if messages['role'] == 'system':
                continue
            else:
                with st.chat_message(messages["role"]):
                    st.markdown(messages["content"])

        if user_message := st.chat_input("Ask me anything AWS-related!"):
            with st.chat_message("user"):
                st.markdown(user_message)
            st.session_state.message.append({"role": "user", "content": user_message})
            chat = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=st.session_state.message,
            )
            response = chat.choices[0].message.content
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.message.append({"role": "assistant", "content": response})