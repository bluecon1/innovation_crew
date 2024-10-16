# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv
# Streamlit
import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from crewai_tools import DirectoryReadTool, \
    FileReadTool, \
    SerperDevTool, \
    DallETool, \
    ScrapeWebsiteTool, \
    WebsiteSearchTool, \
    VisionTool
import streamlit as st
import re
import sys

from openai import api_key

sector = ""
strategic_priorities = ""
key_resource = ""
resources = ""
clients = ""
challenge = ""
openapi_key = ""
model_option = ""

st.title("💬 Innovating with AI Agents!")
with st.sidebar:
    st.header("Enter your inputs below 👇")
    with st.form("my_form1"):
        model_option = st.selectbox(
            "Select the OpenAI Model", ("gpt-4o"))
        openapi_key = st.text_input(
            "Provide your OpenAPI key", type="password")
        # st.write(model_option + openapi_key)
        sector = st.text_input(
            "Provide information on your industry sector", placeholder="B2B vertically integrated coffee manufacturing")
        strategic_priorities = st.text_input(
            "Describe your key strategic priorities",
            placeholder="Identifying desirable and feasible innovations to bring to market")
        key_resource = st.text_input(
            "Kindly input your key resource (e.g., asset)", placeholder="Coffee plants")
        resources = st.text_input(
            "Kindly input your other important resource(s)",
            placeholder="Coffee plantations, coffee manufacturing plants, with all required machinery to extract and package solid and liquid coffee")
        clients = st.text_input(
            "Describe your clients", placeholder="Large fast food chains and coffee retailers")
        challenge = st.text_area(
            "What challenge do you want to solve today?",
            placeholder="Create a list of ideas on using the byproducts of coffee plant and products generated using coffee creation process, broken down by feasibility, desirability and viability and save the file in an .md format. Do include the sources as well for credibility.")

        submitted = st.form_submit_button("Submit")

st.divider()

os.environ["OPENAI_API_KEY"] = openapi_key  # os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = model_option  # os.getenv("OPENAI_MODEL_NAME")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Check if the data contains 'task' information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(
                self.colors)  # Increment color index and wrap around if necessary

            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain",
                                                f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        if "Project Manager" in cleaned_data:
            # Apply different color
            cleaned_data = cleaned_data.replace("Project Manager", f":{self.colors[self.color_index]}[Project Manager]")
        if "Domain Expert" in cleaned_data:
            cleaned_data = cleaned_data.replace("Domain Expert", f":{self.colors[self.color_index]}[Domain Expert]")
        if "Senior Engineer" in cleaned_data:
            # Apply different color
            cleaned_data = cleaned_data.replace("Senior Engineer", f":{self.colors[self.color_index]}[Senior Engineer]")
        if "Senior Marketer" in cleaned_data:
            cleaned_data = cleaned_data.replace("Senior Marketer", f":{self.colors[self.color_index]}[Senior Marketer]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []


# load_dotenv()

if (submitted):
    # Custom file writer function
    # def write_to_file(content, directory='Outputs', filename='output.md'):
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #
    #     file_path = os.path.join(directory, filename)
    #
    #     # Ensure content is a string before writing
    #     if not isinstance(content, str):
    #         content = str(content)
    #
    #     with open(file_path, 'w') as file:
    #         file.write(content)
    #
    #     print(f"Content written to {file_path}")

    # read_inputs = DirectoryReadTool(directory='/content/drive/MyDrive/Creative_AI/research_engine/Inputs')
    # read_outputs = DirectoryReadTool(directory='/content/drive/MyDrive/Creative_AI/research_engine/Outputs')
    # read_baseline = FileReadTool(
    #     file_path='/content/drive/MyDrive/Creative_AI/research_engine/Outputs/knowledge_baseline.txt')
    # file_read_tool = FileReadTool()
    # search_tool = SerperDevTool()
    # scrape_tool = ScrapeWebsiteTool()
    dalle_tool = DallETool(model="dall-e-3",
                           size="1024x1024",
                           quality="standard",
                           n=1)
    website_search_tool = WebsiteSearchTool()

    # vision_tool = VisionTool(model="dall-e-3",
    #                          size="1024x1024",
    #                          quality="standard",
    #                          n=1)

    # Agents
    manager = Agent(
        role="Project Manager",
        goal="Efficiently manage the research team and ensure the production of world-class research reports",
        backstory=(
            "You are a very highly experienced research project manager, and you make sure work is always completed with extremely high standard."
            "You make sure to include multiple revision loops to check the quality and truthfulness of information."
            "Make sure there is first a stage of knowledge collection, then analysis and interpretation, before the report is finalized."
            "Ensure the content is complete, truthful, relevant to the {topic}, {purpose} and {context}."
            "If anything is missing or not at the right level of quality, send it back for revision.\n"
        ),
        # allow_delegation=True,
        verbose=True
    )

    domain_expert = Agent(
        role="Domain Expert",
        goal="Provide domain knowledge on the topics requires, breaking down concepts when needed in effective ways.",
        backstory=(
                "You are a world class domain expert in the sector of " + sector + ". You are particularly knowledgeable on the key resource {key_resource}."
        ),
        # allow_delegation=False,
        verbose=True
    )

    engineer = Agent(
        role="Senior Engineer",
        goal="Identifying the right methods and technologies to perform the tasks required, and assessing their feasibility.",
        backstory=(
                "You are an expert engineer, knowing everything in the world of " + sector + "."
                                                                                             "You are good at providing back of the envelope calculations and estimations on how feasible and complex things are"

        ),
        # allow_delegation=False,
        verbose=True
    )

    marketer = Agent(
        role="Senior Marketer",
        goal="Coming up with clever, original product ideas and assessing how likely people are to buy them.",
        backstory=(
                "You have the pulse of the market for " + sector + "."
                                                                   "You know what clients and consumers want and can rapidly assess whether they will buy it, in what quantities and for what price."
        ),
        # allow_delegation=False,
        verbose=True
    )

    # Tasks
    breakdown_task = Task(
        description=(
                "Break down " + key_resource + " into mutually exclusive, collectively exhaustive sub-components. "
                                               "For example, if tasked with finding alternative uses for coffee production byproducts, "
                                               "you might break it down into beans, trees and coffee grounds. "
                                               "Each of these could be then be broken down into sub-elements, "
                                               "e.g., the tree has root, trunk and leaves."
                                               "Do this based on your training, no need to search online"
                                               "keep in mind the sector, client, resources and challenge for context. \n"

                                               "Generate a table with each part " + key_resource + "(e.g., plant/bean/grounds) in the first column, "
                                                                                                   "the sub-part in the second (e.g., leaves as part of the plant), "
                                                                                                   "the core or active principle in the third (e.g., antioxidants, dietary fiber), "
                                                                                                   "and possible uses or benefits in the fourth. "
                                                                                                   "Search for possible uses online. "
                                                                                                   "Be sure to have one row per each possible use: there should be only one possible use in each row, and multiple rows with possible uses for each active principle and sub-part. \n"

                                                                                                   "Stop and ask the user for confirmation, summarizing what you have done."

                                                                                                   "Sector: " + sector + "\n"
                                                                                                                         "Clients: " + clients + "\n"
                                                                                                                                                 "Challenge: " + challenge + "\n"

        ),
        expected_output=(
            "A table with each part in the first column, "
            "the sub-part in the second (e.g., leaves as part of the plant), "
            "the core or active principle in the third (e.g., antioxidants, dietary fiber), "
            "and possible uses or benefits"
        ),
        tools=[website_search_tool],
        human_input=False,
        agent=domain_expert,
        max_iter=5
    )

    feasibility_task = Task(
        description=(
                "Think step by step. "
                "How can you get from " + key_resource + " to each active principle? "
                                                         "How hard and expensive is it to extract it, for a company with the following resources " + resources + "? \n"
                                                                                                                                                                 "Reason well and add a column for each possible use, with a feasibility score from 1-10."
                                                                                                                                                                 "Keep in mind the sector, client, resources and challenge for context."
                                                                                                                                                                 "Update the table produced by the previous agent with two new columns.\n"

                                                                                                                                                                 "Stop and ask the user for confirmation, summarizing what you have done."

                                                                                                                                                                 "Sector: " + sector + "\n"
                                                                                                                                                                                       "Clients: " + clients + "\n"
                                                                                                                                                                                                               "Challenge: " + challenge + "\n"
                                                                                                                                                                                                                                           "Resources: + " + resources + "\n"

        ),
        expected_output=(
            "An updated version of the table, with two new columns for each possible use, "
            "the rationale for the feasibility of processing or extraction, and a feasibility score from 1-10"
        ),
        # tools=[search_tool, website_search_tool],
        human_input=False,
        agent=engineer,
        max_iter=5
    )

    desirability_task = Task(
        description=(
                "For each possible use, create 3-5 interesting, effective commercial or product ideas. "
                "Make sure that the ideas are original, specific and effective."
                "Then think step by step: "
                "How likely are the target users to buy this product and find it valuable? "
                "How does it compare to the alternatives?"
                "Add for each commercial idea a desirability assessment and a score. "
                "Keep in mind the sector, client, resources and challenge for context."

                "Sector: " + sector + "\n"
                                      "Clients: " + clients + "\n"
                                                              "Challenge: " + challenge + "\n"
                                                                                          "Resources: + " + resources + "\n"

        ),
        expected_output=(
            "An output table with one row for each commercial or product idea, showcasing for each: Part, Sub part, Core/active principle, Possible uses, Difficulty of extraction/application, Feasibility score, Commercial/product ideas, Target market, Desirability assessment, Desirability score, Desirability/feasibility average"
        ),
        tools=[],
        agent=marketer,
        max_iter=5
    )

    prioritize_task = Task(
        description=(
                "Calculate the average between desirability and feasibility scores."
                "Then pick the top 3 ideas, and create a writeup for each, illustrating why their rationale.\n"
                "Keep in mind the sector, client, resources and challenge for context."
                "Create an image for each of the top ideas which illustrates it simply and accurately."

                "Sector: " + sector + "\n"
                                      "Clients: " + clients + "\n"
                                                              "Challenge: " + challenge + "\n"
                                                                                          "Resources: + " + resources + "\n"

        ),
        expected_output=(
            "An output table with one row for each commercial or product idea, showcasing for each: Part, Sub part, Core/active principle, Possible uses, Difficulty of extraction/application, Feasibility score, Commercial/product ideas, Target market, Desirability assessment, Desirability score, Desirability/feasibility average"
        ),
        tools=[dalle_tool],
        agent=domain_expert,
        max_iter=5
    )

    # Initialize the message log in session state if not already present
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to co-innovating with AI Agents."}]

    # Display existing messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Play with planning, process, manager

    crew = Crew(
        agents=[manager, domain_expert, engineer, marketer],
        tasks=[
            breakdown_task,
            desirability_task,
            feasibility_task,
            prioritize_task
        ],

        # process=Process.hierarchical,
        # manager_agent=manager,
        # manager_llm=manager_llm,

        process=Process.hierarchical,
        manager_llm=ChatOpenAI(temperature=0, model=model_option, api_key=openapi_key),
        manager_agent=None,
        planning=True,
        verbose=True,
        memory=True,
        cache=False,
        # share_crew=False,
        # output_log_file="outputs/content_plan_log.txt",
        # max_rpm=50,
        output_name='output1.md'
    )

    # result= crew.kickoff()
    # result = f"Here is the final results \n\n {result}"
    #
    # print("########")
    # print(result)
    # if submitted:
    with st.status("🤖 **Agents at work...**", state="running", expanded=True) as status:
        with st.container(height=500, border=False):
            sys.stdout = StreamToExpander(st)
            final = crew.kickoff()
            # write_to_file(final)
        status.update(label="✅ Innovation Process Complete!",
                      state="complete", expanded=False)

    st.subheader("Here is your consolidated response", anchor=False, divider="rainbow")
    st.markdown(final)
    # Save the report to a file
    # write_to_file(result)