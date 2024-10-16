__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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
domain = ""
# key_resource = ""
# resources = ""
# clients = ""
# challenge = ""
openapi_key = ""
model_option = ""

st.title("💡 Reconverz innovation playground")

# Display the logo in the sidebar
st.sidebar.image('Logo.png', width=80)

with st.sidebar:
    st.header("Enter your inputs below 👇")

    with st.form("my_form1"):
        model_option = st.selectbox(
            "Select the OpenAI Model", ("gpt-4o"))
        openapi_key = st.text_input(
            "Provide your OpenAPI key", type="password")
        # st.write(model_option + openapi_key)
        st.divider()

        st.subheader("Please provide the following inputs:")
        sector = st.text_input(
            "Provide information on your industry sector", 
            value="Insurance",)
        

        # st.subheader("Please specify your domain:")
        domain = st.text_area(
            "Provide information on your domain",
            value="Savings & Retirement")

        submitted = st.form_submit_button("Submit")

        # Add a toggle for verbose output at the bottom of the sidebar
        verbose_toggle = st.checkbox("Enable Verbose Mode for Agents")

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
        if "Web Scouter" in cleaned_data:
            cleaned_data = cleaned_data.replace("Web Scouter", f":{self.colors[self.color_index]}[Web Scouter]")
        if "Persona Text Creator" in cleaned_data:
            # Apply different color
            cleaned_data = cleaned_data.replace("Persona Text Creator", f":{self.colors[self.color_index]}[Persona Text Creator]")
        if "Artist" in cleaned_data:
            cleaned_data = cleaned_data.replace("Artist", f":{self.colors[self.color_index]}[Artist]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []


# load_dotenv()

if (submitted):
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

    verbose_mode = verbose_toggle
    
    manager = Agent(
        role="Project Manager",
        goal="Efficiently manage the research team and ensure the production of world-class customer research reports",
        backstory=(
            "You are a very highly experienced customer research project manager, and you make sure work is always completed with extremely high standard."
            "You make sure to include multiple revision loops to check the quality and truthfulness of information."
            "Make sure there is first a stage of knowledge collection, then analysis and interpretation, before the report is finalized."
            "Ensure the content is complete, truthful, relevant to the " + sector + " sector."
            "If anything is missing or not at the right level of quality, send it back for revision.\n"
        ),
        # allow_delegation=True,
        verbose=verbose_mode
    )

    web_scouter = Agent(
        role="Web Scouter",
        goal="Provide domain knowledge on the topics requires, breaking down concepts when needed in effective ways.",
        backstory=(
                "You are a world class domain expert in the sector of " + sector + ". You are particularly knowledgeable on the domain " + domain + "."
        ),
        # allow_delegation=False,
        verbose=verbose_mode,
        tools=[website_search_tool],

    )

    persona_text_creator = Agent(
        role="Persona Text Creator",
        goal="Analysing data and generating personas to create a true reflection of customer for your " + sector + "sector and specifically within the + " + domain + " domain.",
        backstory=(
                "You are an expert persona generator, knowing everything in the world of " + sector + "."
                "You are good at analysing data and generalising to create personas. "

        ),
        # allow_delegation=False,
        verbose=False,
    )

    artist = Agent(
        role="Artist",
        goal="You use online large language models like OpenAI Dall-E to create awesome images, pertaining to the " + sector + " sector.",
        backstory=(
                "You have the artistic pulse of the market for " + sector + "sector and specifically within the + " + domain + " domain."
                "You have been creating beautiful digital images for a long time now and know how to represent clients' asks on screen."
        ),
        # allow_delegation=False,
        verbose=verbose_mode,
        tools=[dalle_tool],
    )

    # Tasks
    data_gathering_task = Task(
        description=(
                "Search online for companies that operate within the " + sector + " sector and specifically within the " + domain + " domain. "
                "Analyse and gather what type of customers interact with these companies and build a table with 15 personas. "
                "Each persona should have an ID, name and a small description. "
                "For example, this pipe format provides a glimse into how a table row might look like. "
                "1 | Anxious Alex | Alex is in his late 50s, facing a health crisis or disability that's impacting his financial stability."

                "Be sure to have one row per each persona, along with a description. "

        ),
        expected_output=(
            "A table with 3 columns - Persona ID, Persona Name, Persona Description"
        ),
        human_input=False,
        agent=web_scouter,
        max_iter=5,
    )

    persona_text_creator_task = Task(
        description=(
                "Generate 5 mock call transcripts within the " + sector + " sector and specifically within the " + domain + " domain. "
                "For example, if the sector is Insurance and the domain is Savings and Retirement, "
                "you might look at generating some mock data that covers various customer types "
                "such as vulnerable customers, young generation and covers both happy and unhappy scenarios. "
                "Keep in mind the sector and domain for context. \n"
                
                "Now have a look at the table with 15 personas and based on that table and mock data, bring the list down to 5 personas which . "
                "are most relevant to the sector and domain. "
                "Next, for each persona, do the following: "

                "1. Ensure the name is followed by an adjective that summarises this persona well and offers good description. \n" 

                "2. Write a story in a blockquote that uses this template and data from the table: It's really important for me to [GOAL] because I can then [NEED]. The problem I'm trying to fix when it comes to is  [PAIN] but at the moment [FEAR] makes it hard to get there. \n"

                "3. Also, create a table with 2 columns: Problems/Pain Points (what makes the user sad) and Gains (what makes the user happy) with 3 bullets each. \n"

                "4. Lastly, outline key trends and driving forces that will impact the persona in the near future. \n"
        ),
        expected_output=(
            "A textual commentary describing the 5 personas with an innovative name, description, story, goal, pain, need, fear and trends and driving forces. "
        ),
        # tools=[search_tool, website_search_tool],
        human_input=False,
        agent=persona_text_creator,
        max_iter=5
    )

    artist_task = Task(
        description=(
                "Pick the top 5 ideas, and format them to better reading and visualisation. \n"
                "Keep in mind the sector " + sector + " and the domain " + domain + "."
                "Create an image for each of the persona which illustrates it simply and accurately."
                "Make sure the images are simple and tasteful. Ensure the essence of the persona backstory and "
                "description are captured in the generated image"
        ),
        expected_output=(
            "A curated selection of the top 5 personas, comprised of name, description, story, goal, pain, need, fear and trends and "
            " driving forces along with an image."
        ),
        agent=artist,
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
        agents=[web_scouter, persona_text_creator, artist],
        tasks=[
            data_gathering_task,
            persona_text_creator_task,
            artist_task
        ],

        # process=Process.hierarchical,
        # manager_agent=manager,
        # manager_llm=manager_llm,

        process=Process.sequential,
        #manager_llm=ChatOpenAI(temperature=0, model=model_option, api_key=openapi_key),
        #manager_agent=None,
        planning=True,
        verbose=False,
        memory=True,
        cache=False,
        # share_crew=False,
        # output_log_file="outputs/content_plan_log.txt",
        # max_rpm=50,
        # output_name='output1.md'
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