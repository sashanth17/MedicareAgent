from crewai import LLM, Agent, Crew, Task
from tools.medicine import medicine_tool
from tools.BookAppoinment import BookAppointment
# Initialize LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.0  # Deterministic for consistency
)

class CrewInit:
    @staticmethod
    def create_crew():
        """
        Creates a Crew instance with strict formatting rules.
        """

        # ----------------- AGENT -----------------
        agent = Agent(
            llm=llm,
            role="Smart Pharmacy Assistant",
            goal="Provide accurate pharmacy information strictly based on tool outputs.",
            backstory=(
                "You are a precise AI assistant for a pharmacy system. "
                "You value accuracy above all else. "
                "You never add pleasantries or filler text before the final answer."
            ),
            tools=[medicine_tool,BookAppointment],
            max_iter=3,
            verbose=False, # Keep false to reduce noise, though stream=True overrides this often
            allow_delegation=False
        )

        # ----------------- TASK -----------------
        task1 = Task(
            description=(
                f"User Query: '{{query}}'\n\n"
                "STEPS:\n"
                "1. Analyze if the user is asking for medicine availability.\n"
                "2. IF YES: Use 'medicine_tool' with the medicine name.\n"
                "3. IF NO: Answer general questions politely without tools.\n\n"
                "CRITICAL OUTPUT RULES:\n"
                "- You MUST start your final response to the user with 'Final Answer:'.\n"
                "- If the tool finds a pharmacy, your Final Answer must be a complete sentence containing the Name, Location, and Phone.\n"
                "- If the tool returns nothing, state that clearly.\n"
                "- DO NOT include 'Thought:', 'Action:', or raw JSON in your Final Answer."
            ),
            expected_output=(
                "A clear, natural language sentence starting after 'Final Answer:'. "
                "Example: 'Final Answer: Paracetamol is available at Shane Pharma in Coimbatore, contact 143249494.'"
            ),
            agent=agent
        )
        task2 = Task(
    description=(
        f"User Query: '{{query}}'\n\n"
        "STEPS:\n"
        "1. Analyze if the user is asking to book a doctor's appointment.\n"
        "2. IF YES:\n"
        "   - Extract doctor name, patient phone number, appointment date, and optional notes.\n"
        "   - Use 'bookDoctor' tool with extracted details.\n"
        "3. IF NO: Respond politely that you can help with doctor appointment bookings.\n\n"
        "CRITICAL OUTPUT RULES:\n"
        "- You MUST start your final response with: 'Final Answer:'.\n"
        "- If tool booking is successful, clearly present appointment details: Appointment ID, Doctor, Date, Status.\n"
        "- If booking fails or no slots available, clearly state that doctor's appointment is currently not available.\n"
        "- DO NOT include 'Thought:', 'Action:', stack traces, or raw JSON in Final Answer.\n"
        "- Keep responses short, clean, and professional."
    ),
    expected_output=(
        "A formatted appointment confirmation or unavailability message starting after 'Final Answer:'.\n"
        "Example Success: 'Final Answer: Your appointment with Dr. Soundar D on 2025-11-30 is booked successfully. Appointment ID: 1, Appointment Number: 1.'\n"
        "Example Failure: 'Final Answer: Doctor’s appointment is currently not available.'"
    ), 
    agent=agent
    )

        # ----------------- CREW -----------------
        # Note: memory=True appends the JSON reflection block. 
        # Our server.py is now equipped to strip this out automatically.
        crew = Crew(
            agents=[agent],
            tasks=[task1,task2],
            stream=True, 
            memory=True, 
            embedder={
                "provider": "google-generativeai",
                "config": {
                    "model": "gemini-embedding-001",
                    "task_type": "retrieval_document"
                }
            }
        )

        return crew