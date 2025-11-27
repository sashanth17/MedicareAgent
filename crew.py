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
        "You must answer user queries about medicines and pharmacies with HIGH PRECISION. "
        "If the user asks about medicine availability, you MUST call the `medicine_tool` "
        "and then return ONLY the tool's output to the user, with NO extra pharmacies, "
        "NO invented data, and NO modifications to the tool response. "
        "If the user is NOT asking about availability, answer politely without calling the tool."
            ),
            backstory=(
        "You are a helpful pharmacy AI chatbot that must NEVER fabricate medicine or pharmacy data. "
        "When you use tools, you strictly trust and forward their responses without adding or changing "
        "pharmacy names, locations, or contact numbers."
    ),
            tools=[medicine_tool],
            max_iter=2,
            verbose=False,
            show_thoughts=False  # suppress chain-of-thought and internal actions
        )

        # ----------------- TASK -----------------
        task = Task(
            description=(
        "You are chatting with a user. Their message is: {query}\n\n"
        "1. First, decide if the user is asking about MEDICINE AVAILABILITY (for example: "
        "'search paracetomal', 'is paracetamol available', 'where can I buy dolo 650', etc.).\n"
        "2. If the user is asking about availability, you MUST:\n"
        "   - Call `medicine_tool` exactly once with the medicine name.\n"
        "   - Then respond to the user with ONLY the exact string returned by the tool.\n"
        "   - Do NOT summarize, rephrase, re-order, or add ANY extra pharmacies.\n"
        "   - Do NOT hallucinate or guess any data. If the tool returns 2 pharmacies, "
        "     you MUST show exactly those 2 and nothing more.\n"
        "3. If the user is NOT asking about availability, DO NOT call `medicine_tool`. "
        "   Just answer briefly and politely based on your general knowledge.\n"
        "4. You are STRICTLY FORBIDDEN from inventing any pharmacy name, location, or contact number."
    ),
    expected_output=(
        "If `medicine_tool` was called:\n"
        "- The final output MUST be EXACTLY the string returned by `medicine_tool`, "
        "with no additional pharmacies, no fabricated data, and no extra explanation.\n\n"
        "If `medicine_tool` was NOT called:\n"
        "- A short, polite natural-language reply to the user query.\n"
    ),
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