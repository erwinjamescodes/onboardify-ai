import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
import PyPDF2
import time

warnings.filterwarnings("ignore")

# Sidebar for navigation and API key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
openai.api_key = api_key


with st.sidebar:
    page = option_menu(
        "Dashboard",
        ["Home", "About Me", "Onboardify AI"],
        icons=['house', 'info-circle',  'file-text'],
        menu_icon="list",
        default_index=0,
    )

if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to use the application.")

else:
    if page == "Home":
        st.title("ONBOARDIFY AI")
        st.write("Hi! I am Onboardify AI, a web-based tool that generates customized, strategic onboarding experiences for digital products, specifically tailored to the Southeast Asian market, with a focus on the Philippines. This intelligent blueprint generator helps product teams design engaging, user-friendly onboarding processes that maximize user activation and retention!")

        st.write("## How It Works")
        st.write("1. Product Input: Users provide details about their product, including name and brief description.")
        st.write("2. Goal Setting: Select primary onboarding objectives like increasing engagement or reducing user drop-off.")
        st.write("3. Audience Targeting: Define the target user group (e.g., new users, tech-savvy users).")
        st.write("4. Onboarding Structure: Choose the number of onboarding steps and preferred methods.")
        st.write("5. Optional Data Upload: Users can upload user data CSV for more personalized insights.")
        st.write("6. Blueprint Generation: The tool uses AI to create a comprehensive, actionable onboarding strategy.")

        st.write("## Ideal Users")
        st.write("This tool is perfect for:")
        st.write("- UX/UI Designers developing user onboarding experiences")
        st.write("- Growth Hackers focused on user activation")
        st.write("- Startup founders in the Philippines and Southeast Asia")
        st.write("- Digital product teams looking to improve user retention")
        st.write("- Companies with web or mobile applications seeking to optimize first-time user experiences")

        st.markdown("---")
        st.write("Start using Onboardify AI now to solidify your product's onboarding experience!")

    elif page == "About Me":
        st.header("About Me")
        st.markdown("""
        Hi! I'm Erwin Caluag! I am a Software Engineer / Web Developer and an AWS Solutions Architect Associate. Currently, I am venturing into the world of Artificial Intelligence.
                    
        This project is one of the projects I am building to try and apply the learnings I have acquired from the AI First Bootcamp from AI Republic. 
                    
        Any feedback would be greatly appreciated!
        """)

    elif page == "Onboardify AI":
            # def main():
        st.title("Onboardify AI: Personalized Onboarding Blueprint Generator")
        

        st.header("1. Product Description")
        product_name = st.text_input("🏷️ Product Name", placeholder="Enter your product's name")
        
        product_summary = st.text_area(
            "📝 Brief Product Description", 
            placeholder="Provide a concise overview of your product, its main purpose, and key value proposition. (Max 500 characters)",
            height=150,
            max_chars=500
        )
        
        # Character count and validation
        if product_summary:
            chars_remaining = 500 - len(product_summary)
            st.caption(f"Characters remaining: {chars_remaining}")
            
            # Optional sentiment or keyword suggestion
            if len(product_summary) > 50:
                st.caption("💡 Pro Tip: A good product summary clearly explains what problem your product solves")
        
        st.markdown("---")
        # 1. Primary Goal
        st.header("2. What is the primary goal of your onboarding experience?")
        goal_options = [
            "Increase engagement",
            "Reduce drop-off",
            "Showcase key features",
            "Improve user retention"
        ]
        
        selected_goals = []
        for goal in goal_options:
            if st.checkbox(f"📌 {goal}", key=f"goal_{goal}"):
                selected_goals.append(goal)
        
        # Custom goal input
        custom_goal = st.text_input("🔍 Other goal (please specify)")
        if custom_goal:
            selected_goals.append(custom_goal)
            
        # Display selected goals
        if selected_goals:
            st.write("Selected goals:", ", ".join(selected_goals))
        
        st.markdown("---")
        
        # 2. Target Audience
        st.header("3. Who is your target audience?")
        audience_options = [
            "New users (first time using the product)",
            "Tech-savvy users (already familiar with similar products)",
            "First-time users (low experience with technology or your product)",
            "Existing users (users who have used the product previously but need re-engagement)"
        ]
        
        selected_audience = []
        for audience in audience_options:
            if st.checkbox(f"👥 {audience}", key=f"audience_{audience}"):
                selected_audience.append(audience)
        
        # Custom audience input
        custom_audience = st.text_input("🎯 Other audience (please specify)")
        if custom_audience:
            selected_audience.append(custom_audience)
            
        # Display selected audience
        if selected_audience:
            st.write("Selected audience:", ", ".join(selected_audience))
        
        st.markdown("---")
        
        # 3. Number of Steps
        st.header("4. How many steps should your onboarding process include?")
        step_options = ["3-5 steps", "5-10 steps", "10+ steps", "Custom number"]
        selected_steps = st.radio("Select number of steps:", step_options)
        
        if selected_steps == "Custom number":
            custom_steps = st.number_input("Enter custom number of steps:", 
                                        min_value=1, 
                                        max_value=50, 
                                        value=5)
            st.write(f"You selected: {custom_steps} steps")
        else:
            st.write(f"You selected: {selected_steps}")
        
        st.markdown("---")
        
        # 4. Onboarding Methods
        st.header("5. Do you want to incorporate any specific onboarding methods?")
        method_options = [
            "Step-by-step walkthrough",
            "Tooltips (helpful hints throughout the platform)",
            "Self-guided tutorial (user explores independently with minimal guidance)",
            "Video-based onboarding (short instructional videos)"
        ]
        
        selected_methods = []
        for method in method_options:
            if st.checkbox(f"🎓 {method}", key=f"method_{method}"):
                selected_methods.append(method)
        
        # Custom method input
        custom_method = st.text_input("💡 Other method (please specify)")
        if custom_method:
            selected_methods.append(custom_method)
            
        # Display selected methods
        if selected_methods:
            st.write("Selected methods:", ", ".join(selected_methods))
        
        # Generate Blueprint Button
        st.markdown("---")

        st.header("Upload User Data (Optional)")
        uploaded_file = st.file_uploader("Upload a CSV file with user data", type=['csv'])
        
        if uploaded_file is not None:
            try:
                dataframed = pd.read_csv(uploaded_file)
                st.success("File successfully uploaded!")
                
                # Display data preview
                st.subheader("Data Preview")
                st.write("Number of rows:", dataframed.shape[0])
                st.write("Number of columns:", dataframed.shape[1])
                st.dataframe(dataframed.head())
                
            except Exception as e:
                st.error(f"Error reading the file: {str(e)}")
        
        st.markdown("---")

        def generate_blueprint(product_name, product_summary, selected_goals, selected_audience, selected_steps, selected_methods, uploaded_file):
            if uploaded_file is not None:
                try:
                    # Prepare data for embedding
                    dataframed['combined'] = dataframed.apply(lambda row: " ".join(row.values.astype(str)), axis=1)
                    documents = dataframed['combined'].tolist()

                    # Generate embeddings
                    embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in documents]
                    embedding_dim = len(embeddings[0])
                    embeddings_np = np.array(embeddings).astype('float32')
                    index = faiss.IndexFlatL2(embedding_dim)
                    index.add(embeddings_np)

                    # Construct user message
                    user_message = f"""
                    I'm developing a personalized onboarding experience for my product and need a comprehensive onboarding strategy.

                    ## Product Details
                    Product Name: {product_name}
                    Product Description: {product_summary}

                    ## Onboarding Goals
                    {', '.join(selected_goals)}

                    ## Target Audience
                    {', '.join(selected_audience)}

                    ## Onboarding Structure
                    - Number of Steps: {selected_steps}
                    - Onboarding Methods: {', '.join(selected_methods)}

                    ## Context
                    - Target Market: Philippines (73 million internet users)
                    - User Data Insights: {len(documents)} documents analyzed

                    Please generate a detailed, personalized onboarding process that:
                    1. Directly addresses the specified goals
                    2. Tailors the approach to the identified audience
                    3. Follows the preferred onboarding methods
                    4. Creates a clear, engaging user journey
                    5. Provides strategies to maximize user activation

                    Deliver a comprehensive blueprint with specific, actionable recommendations.
                    """

                    # System prompt
                    system_prompt = """
                    You are an expert product onboarding strategist specializing in creating personalized, engaging user experiences for digital products in the Southeast Asian market, with a focus on the Philippines.

                    Core Responsibilities:
                    - Design adaptive onboarding flows that minimize user friction
                    - Create strategies that quickly demonstrate product value
                    - Develop personalized guidance based on user characteristics
                    - Optimize user activation and retention

                    Key Design Principles:
                    - Prioritize intuitive, user-friendly interactions
                    - Use progressive disclosure to manage cognitive load
                    - Incorporate culturally relevant engagement techniques
                    - Align onboarding with specific product goals
                    - Create multiple learning paths for diverse user types

                    Specific Considerations for Philippine Market:
                    - High mobile internet penetration
                    - Young, tech-savvy digital population
                    - Preference for personalized, conversational experiences
                    - Value-driven product interactions
                    - Community and social media influenced decision making

                    Deliverable Requirements:
                    - Provide a step-by-step onboarding flow
                    - Justify each recommended step
                    - Highlight potential user engagement strategies
                    - Suggest methods to reduce user drop-off
                    - Recommend personalization techniques
                    - Define key metrics for measuring onboarding success

                    Tone and Style:
                    - Clear and actionable
                    - Conversational yet professional
                    - Empathetic to user learning curves
                    - Culturally sensitive and inclusive
                    """

                    # Initialize conversation structure
                    struct = [{'role': 'system', 'content': system_prompt}]

                    # Generate and process query embedding
                    query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
                    query_embedding_np = np.array([query_embedding]).astype('float32')
                    _, indices = index.search(query_embedding_np, 5)
                    retrieved_docs = [documents[i] for i in indices[0]]
                    context = ' '.join(retrieved_docs)
                    structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"

                    # Generate response using OpenAI
                    chat = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=struct + [{"role": "user", "content": structured_prompt}],
                        temperature=0.5,
                        max_tokens=1500,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    
                    response = chat.choices[0].message.content
                    
                    # Display results
                    st.success("🎉 Blueprint Generated!")
                    st.markdown("### Your Personalized Onboarding Blueprint")
                    st.markdown(response)

                except Exception as e:
                    st.error(f"Error processing the file: {str(e)}")
            else:
                # Generate blueprint without user data
                system_prompt = """
                You are an expert product onboarding strategist specializing in creating personalized, engaging user experiences for digital products in the Southeast Asian market, with a focus on the Philippines.
                [... rest of the system prompt ...]
                """

                user_message = f"""
                I'm developing a personalized onboarding experience for my product and need a comprehensive onboarding strategy.
                [... rest of the user message without the user data insights ...]
                """

                # Generate response using OpenAI
                chat = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.5,
                    max_tokens=1500,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
                response = chat.choices[0].message.content
                
                # Display results
                st.success("🎉 Blueprint Generated!")
                st.markdown("### Your Personalized Onboarding Blueprint")
                st.markdown(response)

            

        if st.button("Generate Onboarding Blueprint", type="primary"):
            generate_blueprint(product_name, product_summary, selected_goals, selected_audience, selected_steps, selected_methods, uploaded_file)

    
