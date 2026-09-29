#  Lead Generation Agent

A simple AI-powered Lead Generation Agent built using LangGraph, LangChain, Tavily, Ollama, and Streamlit.

This project can search for businesses based on industry and location, collect useful contact information, remove duplicate leads, and generate a final lead report.

## Workflow

The project uses multiple agents/nodes to complete the lead generation process:

1. **Supervisor Agent**
   - Understands the user's request.
   - Decides whether the user wants to start a lead generation task or is simply asking about the agent.
   - Routes the request to the appropriate part of the workflow.

2. **Planner Agent**
   - Extracts the target industry.
   - Extracts the target location.
   - Determines the required number of leads.
   - Generates useful web search queries.

3. **Researcher**
   - Searches the web using Tavily.
   - Collects relevant website URLs and search results.

4. **Lead Extraction Agent**
   - Scrapes website content.
   - Extracts company name, email, phone number, location, and website link.
   - Focuses on the primary business contact information.

5. **Duplicate Removal**
   - Removes duplicate leads based on contact information.
   - Keeps the required number of leads.

6. **Report Generator**
   - Generates a final report.
   - Shows the number of websites searched and scraped.
   - Shows duplicate leads removed.
   - Saves the generated leads as a CSV file.

##  Technologies Used

- Python
- LangGraph
- LangChain
- Ollama
- Tavily
- Streamlit
- Pandas
- Pydantic

Project Files

- `app.py` - Streamlit frontend
- `lead_agent.py` - Backend and LangGraph workflow
- `leadgeneration.ipynb` - Notebook used for learning and experimentation

##  Features

-  Industry and location based lead search
-  Website research and scraping
- Email extraction
-  Phone number extraction
-  Location extraction
-  Website collection
-  CSV report generation
-  LangGraph-based workflow

##  Note

Some websites may fail during scraping because of SSL certificate issues, connection timeouts, or website restrictions. The agent handles these errors and continues processing other websites.
