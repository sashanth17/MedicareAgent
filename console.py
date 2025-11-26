from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, Task
from tools.medicine import medicine_tool
load_dotenv()

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1
)

agent = Agent(
    llm=llm,
    role="Smart Pharmacy Assistant",
    goal="Understand the user query: {query}. Detect if they are asking for medicine availability and call the tool only if needed.",
    backstory="Helpful pharmacy AI chatbot.",
    tools=[medicine_tool],
    max_iter=3  # Prevent long loops
)

task = Task(
    description="Respond politely and make sure to address the user query: {query}. Call the medicine_tool only if they ask about availability.",
    expected_output="Natural conversational streaming output that responds directly to the query.",
    agent=agent
)
embedder_config = {
    "provider": "google-generativeai",
    "config": {
        "model": "gemini-embedding-001",
        "task_type": "retrieval_document"
    }
}
crew = Crew(
    agents=[agent],
    tasks=[task],
    stream=True,
    memory=True,
    embedder=embedder_config
)

print("\n=== Smart Pharmacy Chatbot ===")
print("Type 'exit' to quit\n")
crew.reset_memories(command_type='short')
while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break

    # Send to CrewAI
    streaming = crew.kickoff(inputs={"query": query})

    print("Bot: ", end="", flush=True)
    for chunk in streaming:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")