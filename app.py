import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import textwrap

# Shared search tool
search_tool = DuckDuckGoSearchRun()

# ----------------- Agents -----------------
class TravelAgents:
    def __init__(self):
        self.planner = Agent(
            role='Travel Planner',
            goal='Create detailed travel itineraries between locations',
            backstory='Expert in travel logistics and efficient route planning.',
            tools=[search_tool],
            verbose=True,
            allow_delegation=False
        )
        
        self.researcher = Agent(
            role='Travel Researcher',
            goal='Find transportation options, attractions, and accommodations',
            backstory='Specializes in finding the best travel options and experiences.',
            tools=[search_tool],
            verbose=True,
            allow_delegation=False
        )
        
        self.reviewer = Agent(
            role='Travel Reviewer',
            goal='Ensure the travel plan is comprehensive and practical',
            backstory='Meticulous reviewer ensuring feasibility and completeness.',
            verbose=True,
            allow_delegation=False
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
        return Task(
            description=textwrap.dedent(f"""
                Create a detailed travel itinerary from {self.origin} to {self.destination} on {self.travel_date}.
                Preferences: {self.preferences}.
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
        return Task(
            description=textwrap.dedent(f"""
                Research details for traveling from {self.origin} to {self.destination}.
                Find attractions, accommodations, local transit, cultural tips, and weather.
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
            verbose=2
        )
        
        return crew.kickoff()

# ----------------- Streamlit UI -----------------
st.title("🌍 Travel AI Planner")
st.markdown("Plan your trip with AI-powered travel agents!")

with st.form("travel_form"):
    origin = st.text_input("Where are you traveling from?")
    destination = st.text_input("Where are you traveling to?")
    travel_date = st.text_input("When are you traveling? (YYYY-MM-DD or 'soon')")
    preferences = st.text_area("Any preferences? (budget, luxury, scenic, etc.)")
    
    submitted = st.form_submit_button("Plan My Trip ✈️")

if submitted:
    if travel_date.lower() == "soon":
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
