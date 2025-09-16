import streamlit as st
from crewai import Agent, Task, Crew, Process
from duckduckgo_search import DDGS
from datetime import datetime
import textwrap
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set your Gemini API key (you can get it from: https://aistudio.google.com/app/apikey)
# Option 1: Set it directly (not recommended for production)
GEMINI_API_KEY = "Your api key"  # Replace with your actual API key

# Option 2: Use Streamlit secrets (recommended)
# Create a .streamlit/secrets.toml file and add: GEMINI_API_KEY = "your-api-key"
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    # If not in secrets, use the direct variable (you'll need to set this)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your-gemini-api-key-here")

# Configure Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    verbose=True,
    temperature=0.1,
    google_api_key=GEMINI_API_KEY
)

# Configure the generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Create a search function using DDGS
def search_duckduckgo(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            return "\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search error: {str(e)}"

# ----------------- Agents -----------------
class TravelAgents:
    def __init__(self):
        self.planner = Agent(
            role='Travel Planner',
            goal='Create detailed travel itineraries between locations',
            backstory='Expert in travel logistics and efficient route planning.',
            verbose=True,
            allow_delegation=False,
            llm=llm  # Add Gemini LLM to the agent
        )
        
        self.researcher = Agent(
            role='Travel Researcher',
            goal='Find transportation options, attractions, and accommodations',
            backstory='Specializes in finding the best travel options and experiences.',
            verbose=True,
            allow_delegation=False,
            llm=llm  # Add Gemini LLM to the agent
        )
        
        self.reviewer = Agent(
            role='Travel Reviewer',
            goal='Ensure the travel plan is comprehensive and practical',
            backstory='Meticulous reviewer ensuring feasibility and completeness.',
            verbose=True,
            allow_delegation=False,
            llm=llm  # Add Gemini LLM to the agent
        )

# ----------------- Tasks -----------------
class TravelTasks:
    def __init__(self, origin, destination, travel_date, preferences, agents: TravelAgents):
        self.origin = origin
        self.destination = destination
        self.travel_date = travel_date
        self.preferences = preferences
        self.agents = agents
    
    def plan_itinerary(self):
        # Perform search for transportation options
        transport_query = f"transportation from {self.origin} to {self.destination}"
        transport_info = search_duckduckgo(transport_query)
        
        return Task(
            description=textwrap.dedent(f"""
                Create a detailed travel itinerary from {self.origin} to {self.destination} on {self.travel_date}.
                Preferences: {self.preferences}.
                
                Here's some transportation information I found:
                {transport_info}
                
                Include transportation options, times, costs, and transfers.
            """),
            agent=self.agents.planner,
            expected_output=textwrap.dedent("""
                A travel plan with:
                1. Multiple transportation options
                2. Estimated times and costs
                3. Required transfers or connections
                4. Visa or documentation requirements
            """)
        )
    
    def research_details(self):
        # Perform search for destination details
        attractions_query = f"attractions in {self.destination}"
        attractions_info = search_duckduckgo(attractions_query)
        
        accommodations_query = f"accommodations in {self.destination}"
        accommodations_info = search_duckduckgo(accommodations_query)
        
        return Task(
            description=textwrap.dedent(f"""
                Research details for traveling from {self.origin} to {self.destination}.
                
                Here's some information I found about attractions:
                {attractions_info}
                
                Here's some information about accommodations:
                {accommodations_info}
                
                Find more details about attractions, accommodations, local transit, cultural tips, and weather.
            """),
            agent=self.agents.researcher,
            expected_output=textwrap.dedent("""
                Report with:
                1. Top attractions
                2. 3-5 accommodations with price ranges
                3. Local transit info
                4. Cultural tips or warnings
                5. Weather forecast
            """)
        )
    
    def review_plan(self):
        return Task(
            description=textwrap.dedent(f"""
                Review the complete travel plan from {self.origin} to {self.destination}.
                Ensure it meets preferences, is accurate, and practical.
            """),
            agent=self.agents.reviewer,
            expected_output=textwrap.dedent("""
                Final travel plan with:
                1. Verified transport details
                2. Confirmed attractions & accommodations
                3. Recommendations or warnings
                4. Summary itinerary
            """)
        )

# ----------------- Crew -----------------
class TravelCrew:
    def __init__(self, origin, destination, travel_date, preferences):
        self.origin = origin
        self.destination = destination
        self.travel_date = travel_date
        self.preferences = preferences
        self.agents = TravelAgents()
    
    def run(self):
        tasks = TravelTasks(
            self.origin,
            self.destination,
            self.travel_date,
            self.preferences,
            self.agents
        )
        
        crew = Crew(
            agents=[self.agents.planner, self.agents.researcher, self.agents.reviewer],
            tasks=[tasks.plan_itinerary(), tasks.research_details(), tasks.review_plan()],
            process=Process.sequential,
            verbose=True
        )
        
        return crew.kickoff()

# ----------------- Streamlit UI -----------------
st.title("🌍 Travel AI Planner with Gemini")
st.markdown("Plan your trip with AI-powered travel agents using Google Gemini!")

# API key input (optional - you can set it in secrets or environment variables)
api_key = st.sidebar.text_input("Gemini API Key", type="password", value=GEMINI_API_KEY)
if api_key and api_key != "your-gemini-api-key-here":
    GEMINI_API_KEY = api_key
    # Reinitialize the LLM with the new API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        verbose=True,
        temperature=0.1,
        google_api_key=GEMINI_API_KEY
    )
    genai.configure(api_key=GEMINI_API_KEY)

with st.form("travel_form"):
    origin = st.text_input("Where are you traveling from?")
    destination = st.text_input("Where are you traveling to?")
    travel_date = st.text_input("When are you traveling? (YYYY-MM-DD or 'soon')")
    preferences = st.text_area("Any preferences? (budget, luxury, scenic, etc.)")
    
    submitted = st.form_submit_button("Plan My Trip ✈️")

if submitted:
    if not origin or not destination:
        st.error("Please fill in both origin and destination fields.")
    elif GEMINI_API_KEY == "your-gemini-api-key-here":
        st.error("Please provide a valid Gemini API key in the sidebar.")
    else:
        if travel_date.lower() == "soon" or not travel_date:
            travel_date = datetime.now().strftime("%Y-%m-%d")
        
        st.info("⏳ Planning your trip... please wait.")
        try:
            crew = TravelCrew(origin, destination, travel_date, preferences)
            result = crew.run()
            
            st.success("✅ Your travel plan is ready!")
            st.markdown("### 📌 Complete Itinerary")
            st.markdown(str(result))  # Display as markdown
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
