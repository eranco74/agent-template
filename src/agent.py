import logging
from llama_stack_client import LlamaStackClient
from llama_stack_client import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client import RAGDocument
from llama_stack_client.lib.agents.react.agent import ReActAgent
from llama_stack_client.lib.agents.react.tool_parser import ReActOutput

import sys
import time
import uuid
import os
from dotenv import load_dotenv
from utils import step_logger

load_dotenv()

base_url = os.getenv("REMOTE_BASE_URL")
namespace = os.getenv("NAMESPACE", "llama-serve")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Get a logger instance
logger = logging.getLogger(__name__)

logger.info("Connected to Llama Stack server")

# model_id will later be used to pass the name of the desired inference model to Llama Stack Agents/Inference APIs
model_id = "granite32-8b"
# TODO: remove this
temperature = float(os.getenv("TEMPERATURE", 0.0))
if temperature > 0.0:
    top_p = float(os.getenv("TOP_P", 0.95))
    strategy = {"type": "top_p", "temperature": temperature, "top_p": top_p}
else:
    strategy = {"type": "greedy"}

max_tokens = int(os.getenv("MAX_TOKENS", 512))

# sampling_params will later be used to pass the parameters to Llama Stack Agents/Inference APIs
sampling_params = {
    "strategy": strategy,
    "max_tokens": max_tokens,
}

# For this demo, we are using Milvus Lite, which is our preferred solution. Any other Vector DB supported by Llama Stack can be used.

# RAG vector DB settings
VECTOR_DB_EMBEDDING_MODEL = os.getenv("VDB_EMBEDDING")
VECTOR_DB_EMBEDDING_DIMENSION = int(os.getenv("VDB_EMBEDDING_DIMENSION", 384))
VECTOR_DB_CHUNK_SIZE = int(os.getenv("VECTOR_DB_CHUNK_SIZE", 512))
VECTOR_DB_PROVIDER_ID = os.getenv("VDB_PROVIDER")

# Unique DB ID for session
vector_db_id = f"test_vector_db_{uuid.uuid4()}"

logger.info(f"Inference Parameters:\tModel: {model_id}\tSampling Parameters: {sampling_params}")

# Optional: Enter your MCP server URL here
ocp_mcp_url = os.getenv("REMOTE_OCP_MCP_URL") # Optional: enter your MCP server url here
slack_mcp_url = os.getenv("REMOTE_SLACK_MCP_URL") # Optional: enter your MCP server url here


def create_client()-> LlamaStackClient:
    """
    Create a LlamaStackClient instance and register necessary tool groups.
    This function checks if the required tool groups are already registered,
    and if not, registers them with the Llama Stack server.
    Returns:
        LlamaStackClient: An instance of the LlamaStackClient connected to the specified base URL.
    """
    client = LlamaStackClient(
        base_url=base_url,
    )
    # Get list of registered tools and extract their toolgroup IDs
    registered_tools = client.tools.list()
    registered_toolgroups = [tool.toolgroup_id for tool in registered_tools]

    if  "builtin::rag" not in registered_toolgroups: # Required
        client.toolgroups.register(
            toolgroup_id="builtin::rag",
            provider_id="milvus"
        )

    if "mcp::openshift" not in registered_toolgroups: # required
        client.toolgroups.register(
            toolgroup_id="mcp::openshift",
            provider_id="model-context-protocol",
            mcp_endpoint={"uri":ocp_mcp_url},
        )

    if "mcp::slack" not in registered_toolgroups: # required
        client.toolgroups.register(
            toolgroup_id="mcp::slack",
            provider_id="model-context-protocol",
            mcp_endpoint={"uri":slack_mcp_url},
        )

    # Log the current toolgroups registered
    logger.info(f"Your Llama Stack server is already registered with the following tool groups: {set(registered_toolgroups)}\n")
    return client

def define_rag(client: LlamaStackClient):
    """
    Define and register a document collection for RAG (Retrieval-Augmented Generation) with the Llama Stack server.
    This function creates a vector database for storing documents and ingests a openshift container platform support document into it.
    Args:
        client (LlamaStackClient): An instance of the LlamaStackClient connected to the Llama Stack server.
    """
    # define and register the document collection to be used
    client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=VECTOR_DB_EMBEDDING_MODEL,
        embedding_dimension=VECTOR_DB_EMBEDDING_DIMENSION,
        provider_id=VECTOR_DB_PROVIDER_ID,
    )

    # ingest the documents into the newly created document collection
    urls = [
        ("https://docs.redhat.com/en/documentation/openshift_container_platform/4.18/pdf/support/index", "application/pdf"),
    ]
    documents = [
        RAGDocument(
            document_id=f"num-{i}",
            content=url,
            mime_type=url_type,
            metadata={},
        )
        for i, (url, url_type) in enumerate(urls)
    ]
    client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=VECTOR_DB_CHUNK_SIZE,
    )

def create_agent(client: LlamaStackClient) -> Agent:
    """
    Create an agent with the specified model and tools.
    Args:
        client (LlamaStackClient): An instance of the LlamaStackClient connected to the Llama Stack server.
    """
    # Define the agent's system prompt
    model_prompt= """You are a helpful assistant. You have access to a number of tools.
Whenever a tool is called, be sure return the Response in a friendly and helpful tone."""
    # Create simple agent with tools
    agent = Agent(
        client,
        model=model_id, # replace this with your choice of model
        instructions = model_prompt , # update system prompt based on the model you are using
        tools=[dict(
                name="builtin::rag",
                args={
                    "vector_db_ids": [vector_db_id],  # list of IDs of document collections to consider during retrieval
                },
            ),"mcp::openshift", "mcp::slack"],
        tool_config={"tool_choice":"auto"},
        sampling_params=sampling_params
    )
    return agent



def run_agent_monitoring(agent_instance, target_namespace, use_stream=False):
    """
    Triggers the agent to perform its monitoring tasks for all pods in the namespace,
    with internal logic for preventing duplicate Slack notifications.
    Args:
        agent_instance (Agent): The agent instance to use for monitoring.
        target_namespace (str): The OpenShift namespace to monitor.
        use_stream (bool): Whether to stream the agent's response in real-time.
    """
    session_id = agent_instance.create_session(session_name=f"OCP_Slack_Pod_Monitor_{int(time.time())}")

    # The prompt should be comprehensive, telling the agent what to do,
    # and crucially, that it should only notify on *new* errors.
    # The agent's internal tools/memory will handle the "if already notified" logic.

    logger.info(f"Triggering agent for pod monitoring in namespace '{target_namespace}' at {time.ctime()}...")

    user_prompts = [
        f"list the pods in {target_namespace} OpenShift namespace, if the pod status is in error state check the conditions and pod logs.",
        "Search for solutions on this error and provide a summary of the steps to take in just 1-2 sentences.",
        """Send a Slack message to the 'demos' channel for each pod that has an error summary.
        # Use this format:
        # "⚠️ Pod '{pod-name}' is in an error state. {summary}"
        """
    ]
    session_id = agent.create_session(session_name="OCP_Slack_demo")
    for i, prompt in enumerate(user_prompts):
        response = agent_instance.create_turn(
            messages=[
                {
                    "role":"user",
                    "content": prompt
                }
            ],
            session_id=session_id,
            stream=use_stream,
        )
        step_logger(response.steps)

    logger.info("Agent monitoring cycle completed.")

if __name__ == "__main__":
    client = create_client()
    define_rag(client)
    agent = create_agent(client)
    while True:
        try:
            run_agent_monitoring(agent, namespace) # Pass your agent instance and namespace
        except Exception as e:
            print(f"An error occurred during agent monitoring: {e}")
            # Implement more robust error handling if needed, e.g., logging to a file

        print(f"Waiting for 1 minute before next check... (Next check at {time.ctime(time.time() + 60)})")
        time.sleep(60) # Wait for 60 seconds (1 minute)
