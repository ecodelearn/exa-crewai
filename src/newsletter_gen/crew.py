from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from newsletter_gen.tools.research import SearchAndContents, FindSimilar, GetContents
from datetime import datetime
import streamlit as st
from typing import Union, List, Tuple, Dict
from langchain_core.agents import AgentFinish
import json
import yaml
import os
from langchain_openai import ChatOpenAI

# Função para carregar a configuração dos LLMs e a chave da API da Exa
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config/llm_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Carregar a configuração dos LLMs
config = load_config()
llms = config['llms']
exa_api_key = config.get('exa_api_key')

# Definir a chave da API da Exa como variável de ambiente, se não estiver definida
if exa_api_key:
    os.environ['EXA_API_KEY'] = exa_api_key

# Função para obter o LLM pelo id
def get_llm_by_id(llm_id):
    for llm_config in llms:
        if llm_config['id'] == llm_id:
            return ChatOpenAI(model_name=llm_config['model_name'], api_key=llm_config['api_key'])
    raise ValueError(f"LLM com id {llm_id} não suportado")

@CrewBase
class NewsletterGenCrew:
    """NewsletterGen crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def step_callback(
        self,
        agent_output: Union[str, List[Tuple[Dict, str]], AgentFinish],
        agent_name,
        *args,
    ):
        with st.chat_message("AI"):
            # Try to parse the output if it is a JSON string
            if isinstance(agent_output, str):
                try:
                    agent_output = json.loads(agent_output)
                except json.JSONDecodeError:
                    pass

            if isinstance(agent_output, list) and all(
                isinstance(item, tuple) for item in agent_output
            ):
                for action, description in agent_output:
                    # Print attributes based on assumed structure
                    st.write(f"Agent Name: {agent_name}")
                    st.write(f"Tool used: {getattr(action, 'tool', 'Unknown')}")
                    st.write(f"Tool input: {getattr(action, 'tool_input', 'Unknown')}")
                    st.write(f"{getattr(action, 'log', 'Unknown')}")
                    with st.expander("Show observation"):
                        st.markdown(f"Observation\n\n{description}")

            # Check if the output is a dictionary as in the second case
            elif isinstance(agent_output, AgentFinish):
                st.write(f"Agent Name: {agent_name}")
                output = agent_output.return_values
                st.write(f"I finished my task:\n{output['output']}")

            # Handle unexpected formats
            else:
                st.write(type(agent_output))
                st.write(agent_output)

    @agent
    def researcher(self) -> Agent:
        llm = get_llm_by_id("openai")
        return Agent(
            config=self.agents_config["researcher"],
            tools=[SearchAndContents(api_key=exa_api_key), FindSimilar(), GetContents()],
            verbose=True,
            llm=llm,
            step_callback=lambda step: self.step_callback(step, "Research Agent"),
        )

    @agent
    def editor(self) -> Agent:
        llm = get_llm_by_id("openai")
        return Agent(
            config=self.agents_config["editor"],
            verbose=True,
            tools=[SearchAndContents(api_key=exa_api_key), FindSimilar(), GetContents()],
            llm=llm,
            step_callback=lambda step: self.step_callback(step, "Chief Editor"),
        )

    @agent
    def designer(self) -> Agent:
        llm = get_llm_by_id("openai")
        return Agent(
            config=self.agents_config["designer"],
            verbose=True,
            allow_delegation=False,
            llm=llm,
            step_callback=lambda step: self.step_callback(step, "HTML Writer"),
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher(),
            output_file=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_research_task.md",
        )

    @task
    def edit_task(self) -> Task:
        return Task(
            config=self.tasks_config["edit_task"],
            agent=self.editor(),
            output_file=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_edit_task.md",
        )

    @task
    def newsletter_task(self) -> Task:
        return Task(
            config=self.tasks_config["newsletter_task"],
            agent=self.designer(),
            output_file=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_newsletter_task.html",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the NewsletterGen crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=2,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

# Executando a crew
if __name__ == "__main__":
    inputs = {"topic": "AI in healthcare"}
    NewsletterGenCrew().crew().kickoff(inputs=inputs)
