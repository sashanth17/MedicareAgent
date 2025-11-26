from crewai import LLM, Agent, Crew, Task
from tools.medicine import medicine_tool

# Initialize LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.0  # deterministic responses
)

class CrewInit:
    @staticmethod
    def create_crew():
        """
        Create a Crew instance per connection.
        Ensures per-client memory isolation and strict user-facing responses.
        """

        # ----------------- AGENT -----------------
        agent = Agent(
            llm=llm,
            role="Smart Pharmacy Assistant",
            goal=(
                "You are a polite and concise pharmacy assistant. "
                "Answer only the user query. "
                "Never mention tools, actions, thoughts, JSON, or debug info. "
                "For medicine availability, list pharmacies in bullet points with name, location, and contact number."
            ),
            backstory="Helpful pharmacy AI. Output must be human-readable only, no internal reasoning.",
            tools=[medicine_tool],
            max_iter=2,
            verbose=False,
            show_thoughts=False  # suppress chain-of-thought and internal actions
        )

        # ----------------- TASK -----------------
        task = Task(
            description=(
                "Respond to the user query: {query}. "
                "Use tools only if needed. Output must be clean, human-readable text only."
            ),
            expected_output="Streaming human-friendly answer.",
            agent=agent
        )

        # ----------------- EMBEDDER -----------------
        embedder_config = {
            "provider": "google-generativeai",
            "config": {
                "model": "gemini-embedding-001",
                "task_type": "retrieval_document"
            }
        }

        # ----------------- CREW -----------------
        crew = Crew(
            agents=[agent],
            tasks=[task],
            stream=True,
            memory=True,
            embedder=embedder_config
        )

        return crew