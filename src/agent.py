import os
import logging
from tqdm import tqdm
import yaml
from dotenv import load_dotenv
import wikipedia
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

# Load configuration
with open("../config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Setup logging
if not os.path.exists(CONFIG["logs_folder"]):
    os.makedirs(CONFIG["logs_folder"])
logging.basicConfig(
    filename=os.path.join(CONFIG["logs_folder"], CONFIG["log_file"]),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize memory
memory = ConversationBufferMemory(
    max_length=CONFIG["memory_length"], memory_key="chat_history"
)

# Initialize model
llm = ChatOpenAI(temperature=0)

# Load tools
tools = load_tools(["serpapi", "wikipedia", "llm-math"], llm=llm)

# Initialize the agent with memory
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=True,
    verbose=True,
)

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    response = agent.run(input=user_input)
    print(f"Bot: {response}")
