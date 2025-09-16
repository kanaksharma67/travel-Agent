from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import textwrap


search_tool = DuckDuckGoSearchRun()

class TravelAgents:
    def __init__(self):
        self.planner = Agent(
            role='Travel Planner',
            goal='Create detailed travel itineraries between locations',
            backstory='An expert in travel logistics and route planning with years of experience in creating efficient travel routes.',
            tools=[search_tool],
            verbose=True,
            allow_delegation=False
        )
        
        self.researcher = Agent(
            role='Travel Researcher',
            goal='Find the best transportation options, attractions, and accommodations',
            backstory='A knowledgeable researcher who specializes in finding the best travel options and points of interest.',
            tools=[search_tool],
            verbose=True,
            allow_delegation=False
        )
        
        self.reviewer = Agent(
            role='Travel Reviewer',
            goal='Ensure the travel plan is comprehensive and practical',
            backstory='A meticulous reviewer who double-checks all travel plans for completeness and feasibility.',
            verbose=True,
            allow_delegation=False
        )

class TravelTasks:
    def __init__(self, origin, destination, travel_date, preferences):
        self.origin = origin
        self.destination = destination
        self.travel_date = travel_date
        self.preferences = preferences
    
    def plan_itinerary(self):
        return Task(
            description=textwrap.dedent(f"""
                Create a detailed travel itinerary from {self.origin} to {self.destination} for date {self.travel_date}.
                Consider the following preferences: {self.preferences}.
                Include transportation options, estimated times, costs, and any necessary transfers.
            """),
            agent=TravelAgents().planner,
            expected_output=textwrap.dedent("""
                A comprehensive travel plan with:
                1. Multiple transportation options (flight, train, bus, etc.)
                2. Estimated travel times for each option
                3. Approximate costs
                4. Required transfers or connections
                5. Any visa or documentation requirements
            """)
        )
    
    def research_details(self):
        return Task(
            description=textwrap.dedent(f"""
                Research additional details for traveling from {self.origin} to {self.destination}.
                Find:
                1. Top attractions at the destination
                2. Recommended accommodations
                3. Local transportation options
                4. Cultural tips or important notices
                5. Weather forecast for the travel date
            """),
            agent=TravelAgents().researcher,
            expected_output=textwrap.dedent("""
                A detailed report containing:
                1. List of top attractions with brief descriptions
                2. 3-5 accommodation options with price ranges
                3. Local transit information
                4. Important cultural tips or warnings
                5. Expected weather conditions
            """)
        )
    
    def review_plan(self):
        return Task(
            description=textwrap.dedent(f"""
                Review the complete travel plan from {self.origin} to {self.destination}.
                Ensure all information is accurate, practical, and meets the traveler's preferences.
                Identify any potential issues or missing information.
            """),
            agent=TravelAgents().reviewer,
            expected_output=textwrap.dedent("""
                A finalized travel plan that includes:
                1. Verified transportation details
                2. Confirmed attraction and accommodation information
                3. Any additional recommendations or warnings
                4. A summary of the complete itinerary
            """)
        )

# Main Crew
class TravelCrew:
    def __init__(self, origin, destination, travel_date, preferences):
        self.origin = origin
        self.destination = destination
        self.travel_date = travel_date
        self.preferences = preferences
    
    def run(self):
        tasks = TravelTasks(
            self.origin,
            self.destination,
            self.travel_date,
            self.preferences
        )
        
        itinerary_task = tasks.plan_itinerary()
        research_task = tasks.research_details()
        review_task = tasks.review_plan()
        
        crew = Crew(
            agents=[TravelAgents().planner, TravelAgents().researcher, TravelAgents().reviewer],
            tasks=[itinerary_task, research_task, review_task],
            process=Process.sequential,
            verbose=2
        )
        
        result = crew.kickoff()
        return result

# CLI Interface
def main():
    print("\n Travel AI Planner ")
    print("I'll help you plan your trip from anywhere to anywhere!\n")
    
    origin = input("Where are you traveling from? ")
    destination = input("Where are you traveling to? ")
    travel_date = input("When are you traveling? (YYYY-MM-DD or 'soon') ")
    preferences = input("Any special preferences? (budget, luxury, fast, scenic, etc.) ")
    
    if travel_date.lower() == 'soon':
        travel_date = datetime.now().strftime("%Y-%m-%d")
    
    print("\n Planning your trip... This may take a moment.\n")
    
    try:
        crew = TravelCrew(origin, destination, travel_date, preferences)
        result = crew.run()
        
        print("\n Here's your complete travel plan:\n")
        print(result)
        print("\nHave a great trip! ")
        
    except Exception as e:
        print(f"\n Error occurred: {e}")
        print("Please try again with different inputs or check your internet connection.")

if __name__ == "__main__":
    main()



#pip install crewai langchain_community duckduckgo-search google-generativeai python-dotenv
