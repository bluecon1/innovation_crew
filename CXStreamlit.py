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

load_dotenv
print(1)

# file = open('data.txt','r')
# contents = file.read()
# file.close()

print(2)
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

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_MODEL_NAME"] = os.getenv("OPENAI_MODEL_NAME")

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
        if "Persona Creator" in cleaned_data:
            cleaned_data = cleaned_data.replace("Domain Expert", f":{self.colors[self.color_index]}[Domain Expert]")
        if "Prompt Assistant" in cleaned_data:
            # Apply different color
            cleaned_data = cleaned_data.replace("Senior Engineer", f":{self.colors[self.color_index]}[Senior Engineer]")
        if "Artist" in cleaned_data:
            cleaned_data = cleaned_data.replace("Senior Marketer", f":{self.colors[self.color_index]}[Senior Marketer]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []


# load_dotenv()
print(3)
if (submitted):
    print(4)
    dalle_tool = DallETool(model="dall-e-3",
                           size="1024x1024",
                           quality="standard",
                           n=1)
    website_search_tool = WebsiteSearchTool()

    # Agents
    agent_manager = Agent(
        role="Project Manager",
        goal="Efficiently manage the customer research team and ensure the production of world-class customer research outputs such as personas",
        backstory=(
            "You are a very highly experienced customer research project manager, and you make sure work is always completed with extremely high standard."
            "You make sure to include multiple revision loops to check the quality and truthfulness of information."
            "Make sure there is first a stage of knowledge collection, then analysis and interpretation, before the output is finalized."
            "Ensure the content is complete, truthful, relevant to the customer research."
            "If anything is missing or not at the right level of quality, send it back for revision.\n"
        ),
        # allow_delegation=True,
        verbose=True
    )

    agent_persona_creator = Agent(
        role="Persona Creator",
        goal="You are a user researcher for a project trying to design better customer experience flows for customers calling into contact centre regarding their pensions.",
        backstory=(
                "You are a world class creative customer researcher in the Insurance sector. You have more than 10 years of expertise in analysing data and creating personas."
        ),
        # allow_delegation=False,
        verbose=True
    )

    agent_prompt_assistant = Agent(
        role="Prompt Assistant",
        goal="You are the master in creating specific and subjective prompts that return the best output.",
        backstory=(
                "You are an expert prompt engineer with more than 5 years in crafting prompts within Customer Research domain."

        ),
        # allow_delegation=False,
        verbose=True
    )

    agent_artist = Agent(
        role="Artist",
        goal="You are the creative arm of this team, generating beautiful and relevant images using OpenAI Dall-E.",
        backstory=(
                "You have been creating digital art for over 7 years, with a passion for customer excellence."
        ),
        # allow_delegation=False,
        verbose=True
    )

    # Tasks
    task_persona = Task(
        description=(
            '''
         The transcript data is here:''' + contents +

        '''Absorb all the transcripts and create 3 personas based on this knowledge.  

        For each persona, do the following:
        Invent a name followed by an adjective that summarises this persona well and offers description. 
        Write a story in a blockquote that uses this template: It's really important for me to [GOAL] because I can then [NEED]. The problem I'm trying to fix when it comes to is  [PAIN] but at the moment [FEAR] makes it hard to get there. 
        Also, create a table with 2 columns: Problems/Pain Points (what makes the user sad) and Gains (what makes the user happy) with 3 bullets each.

        Lastly, outline key trends and driving forces that will impact the persona in the near future.
        ''',
        ),
        
        expected_output=(
            "A compelling and summarised textual output for 3 personas"
        ),
        # tools=[website_search_tool],
        human_input=False,
        agent=agent_persona_creator,
        max_iter=5
    )

    task_prompt_assistant = Task(
        description=(
                '''Your goal is to generate creative and accurate 3 image prompts for the model based on 3 persona inputs from Persona_Creator.

                **Your Responsibilities:**

                1. **Analyze User Input:** Carefully read the Persona_Creator's messages and identify their desired image 
                2. **Understand User Preferences:** Determine the user's preferred style, tone, and overall aesthetic (e.g., realistic, cartoon, abstract, whimsical).
                3. **Generate Image Prompts:** Craft one or more detailed image generation prompts based on the user's message and preferences. 
                    - **Clarity is Key:** Make sure your prompts are clear, specific, and easy for the DALL-E model to interpret.
                    - **Include Details:** Provide information about subject, setting, action, style, and composition. 
                    - **Use Descriptive Language:**  Choose words and phrases that evoke the desired visual style and imagery.

                Please make sure your response only contains prompt sentences, without including any description or introduction of this prompt.
                '''
        ),
        expected_output=(
            "Textual output"
        ),
        # tools=[search_tool, website_search_tool],
        human_input=False,
        agent=agent_prompt_assistant,
        max_iter=5
    )

    task_artist = Task(
        description=(
                "Generate images based on the 3 personas."
        ),
        expected_output=(
            "An output with persona details - text and image for each of the persona."
        ),
        tools=[dalle_tool],
        agent=agent_artist,
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
        agents=[agent_persona_creator, agent_prompt_assistant, agent_artist],
        tasks=[task_persona, task_prompt_assistant, task_artist],

        # process=Process.hierarchical,
        # manager_agent=manager,
        # manager_llm=manager_llm,

        process=Process.sequential,
        # manager_llm=ChatOpenAI(temperature=0, model=model_option, api_key=openapi_key),
        # manager_agent=agent_manager,
        # planning=True,
        verbose=True,
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