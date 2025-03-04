import streamlit as st
import asyncio
import aiohttp
import feedparser
import re
import urllib.parse
from bs4 import BeautifulSoup
import io
import PyPDF2
from agentjo import strict_json_async  # Assuming agentjo is pip installable
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib import colors


# --- Utility Functions ---
async def llm(system_prompt: str, user_prompt: str) -> str:
    ''' Here, we use OpenAI for illustration, you can change it to your own LLM '''
    # ensure your LLM imports are all within this function
    from openai import AsyncOpenAI

    # define your own LLM here
    client = AsyncOpenAI(api_key=st.session_state.openai_api_key)
    response = await client.chat.completions.create(
        model=st.session_state.llm_model_name,
        # temperature = 0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def format_arxiv_query(query):
    words = query.strip().split()
    if not words:
        return ""
    formatted_query = " AND ".join(f"all:{word}" for word in words)
    return urllib.parse.quote(formatted_query)

def extract_citation_key(bibtex_text):
    pattern = r'@\w+\s*\{\s*([^,]+),'
    match = re.search(pattern, bibtex_text)
    if match:
        return match.group(1).strip()
    return None

async def async_fetch_feed(query_url, session):
    async with session.get(query_url) as response:
        response.raise_for_status()
        return await response.text()

async def async_get_bibtex_entry(entry, session):
    try:
        abs_url = entry.id
        parts = abs_url.split('/abs/')
        if len(parts) < 2:
            return "No valid arXiv id found in entry.id"
        arxiv_id_with_version = parts[1]
        arxiv_id = re.split("v", arxiv_id_with_version)[0]
        bibtex_url = f"https://arxiv.org/bibtex/{arxiv_id}"
        async with session.get(bibtex_url) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        return f"Error retrieving BibTeX: {str(e)}"

async def async_extract_pdf_text(pdf_url, session):
    try:
        async with session.get(pdf_url) as response:
            response.raise_for_status()
            content = await response.read()

            def extract_text(content):
                with io.BytesIO(content) as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    return text if text else "No text could be extracted from the PDF."

            return await asyncio.to_thread(extract_text, content)
    except Exception as e:
        return f"Error retrieving PDF text: {str(e)}"

async def async_extract_html_text(html_url, session):
    try:
        async with session.get(html_url) as response:
            response.raise_for_status()
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator="\n")
            cleaned_text = re.sub(r'\n+', '\n', text).strip()
            return cleaned_text
    except Exception as e:
        return f"Error retrieving HTML text: {str(e)}"

async def async_retrieve_important(user_output: str, metadata_text: str):
    '''Retrieve what is important to user_output from metadata_text'''
    res = await strict_json_async(f'''From the Text, extract useful information for query: ```{user_output}```
You must put all details so that another person can understand without referencing the Text.
You must output quantitative results and detailed descriptions whenever applicable.
You must output 'NA' if Text is not useful for query or if you are unsure''',
                             "Text: " + metadata_text[:200000],
                             output_format = {
                                 "Text Relevant for query": "type: bool",
                                 "Important Information": "type: str",
                                 "Filtered Detailed Important Information": f"Be detailed, only those directly related to query ```{user_output}```, 'NA' if not useful, type: str"},
                             llm = llm)
    if res["Text Relevant for query"]:
        return res["Filtered Detailed Important Information"]
    else:
        return 'NA'

async def search_arxiv(query, user_output, session):
    base_url = "http://export.arxiv.org/api/query?"
    formatted_query = format_arxiv_query(query)
    query_url = f"{base_url}search_query={formatted_query}&start=0&max_results=10"

    # Fetch the arXiv feed asynchronously.
    xml_data = await async_fetch_feed(query_url, session)
    feed = feedparser.parse(xml_data)

    # Schedule tasks for retrieving BibTeX entries and metadata concurrently.
    bibtex_tasks = []
    metadata_tasks = []
    for entry in feed.entries:
        # BibTeX task
        bibtex_tasks.append(async_get_bibtex_entry(entry, session))

        # Determine PDF and HTML URLs.
        pdf_url = None
        html_url = None
        if hasattr(entry, 'links'):
            for link in entry.links:
                if link.get('type') == 'application/pdf':
                    pdf_url = link.href
                elif link.get('rel') == 'alternate':
                    html_url = link.href

        # Metadata task: try PDF first, then HTML.
        if pdf_url:
            metadata_tasks.append(async_extract_pdf_text(pdf_url, session))
        elif html_url:
            metadata_tasks.append(async_extract_html_text(html_url, session))
        else:
            # If no PDF or HTML link is available, return a default message.
            metadata_tasks.append(asyncio.sleep(0, result = entry.summary.strip()))

    # Await all BibTeX and metadata tasks concurrently.
    bibtex_entries = await asyncio.gather(*bibtex_tasks)
    metadata_texts = await asyncio.gather(*metadata_tasks)

    # make bibtex_entries into dict form
    bibtex_dict = {extract_citation_key(bibtex_entry): bibtex_entry for bibtex_entry in bibtex_entries}

    # Get the important information out
    synthesis_tasks = [async_retrieve_important(user_output, metadata_texts[i]) for i in range(len(metadata_texts))]
    important_information = await asyncio.gather(*synthesis_tasks)

    return bibtex_dict, important_information



async def generate_report_markdown(bibtex_dict, important_information, user_output):
    # Further corrected f-string syntax
    res = await strict_json_async(f'''Generate a research report in markdown format for the query: ```{user_output}```
If format is specified, follow format strictly.
You must do in-line citation with the [[1]], [[2]], [[3]] ... whenever possible.
Link the citation url in the [[1]], [[2]], [[3]]
Use as many sources as possible for each section of the report
At the end of the report, list out all the sources using:
`[source_number]: APA citation`

Citation Details: ```{bibtex_dict}```''',
            important_information,
            output_format = {"Research Report": "Include citations, be as detailed as possible, type: str"},
            llm = llm)

    report = res["Research Report"]
    return report


def generate_pdf_from_markdown(markdown_text, bibtex_dict):
    """Generates a PDF from Markdown text using ReportLab."""

    doc = SimpleDocTemplate("report.pdf", pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Heading1',
                            parent=styles['Heading1'],
                            alignment=TA_CENTER,
                            spaceAfter=12))

    styles.add(ParagraphStyle(name='Normal_CENTER',
                          parent=styles['Normal'],
                          alignment=TA_CENTER,
                          ))

    story = []

    # Split the Markdown text into paragraphs
    paragraphs = markdown_text.split("\n\n")

    for paragraph in paragraphs:
        # Simple heading detection (you might need a more robust method)
        if paragraph.startswith("# "):
            story.append(Paragraph(paragraph[2:], styles['Heading1']))
        elif paragraph.startswith("## "):
            story.append(Paragraph(paragraph[3:], styles['Heading2']))
        elif paragraph.startswith("### "):
            story.append(Paragraph(paragraph[4:], styles['Heading3']))
        elif paragraph.startswith("---"):
            story.append(PageBreak())
        else:
             # Replace Markdown links with ReportLab's <link> tag
            paragraph = re.sub(r'\[\[(.*?)\]\]\((.*?)\)', r'<link href="\2">\1</link>', paragraph)
            story.append(Paragraph(paragraph, styles['Justify']))
        story.append(Spacer(1, 12))

    doc.build(story)
    with open("report.pdf", "rb") as f:
        pdf_data = f.read()
    return pdf_data



# --- Streamlit App ---

async def main():  # Make main async
    st.title("Deep Research arXiv Paper Summarizer")

    # Input for OpenAI API Key
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password", value=st.session_state.openai_api_key)
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key

    available_reasoning_model_list = ["gpt-4o", "o1", "o1-preview", "o1-mini", "o3-mini"]
    if "llm_model_name" not in st.session_state:
        st.session_state.llm_model_name = available_reasoning_model_list[0]
    llm_model_name = st.selectbox("Select the reasoning model LLM Model Name", available_reasoning_model_list)
    st.session_state.llm_model_name = llm_model_name

    # Input for arXiv Search Query
    user_query = st.text_input("Enter your search query for arXiv papers:", "memory adaptive neuroscience")

    # Input for User Output Format (can be a file upload)
    user_output = st.text_area("Enter the desired output format:", '''What is memory?
    Required output format:
    1. Introduction
    2. Types of Memory
    3. How Memory can be adaptive
    4. How Memory schema is created
    5. Future focus areas on memory
    6. Conclusion''')



    # File Uploader for Custom Prompts (optional)
    uploaded_file = st.file_uploader("Upload a custom prompt file (optional)", type=["txt"])
    if uploaded_file is not None:
        string_data = uploaded_file.read().decode("utf-8")
        user_output = string_data  # Override user_output with file content


    if st.button("Generate Report"):
        if not st.session_state.openai_api_key:
            st.warning("Please enter your OpenAI API Key.")
            return

        with st.spinner("Searching arXiv and generating report..."):
            try:
                async with aiohttp.ClientSession() as session:
                    bibtex_dict, important_information = await search_arxiv(user_query, user_output, session)
                    report_md = await generate_report_markdown(bibtex_dict, important_information, user_output)

                st.markdown(report_md)  # Display Markdown
                pdf_data = generate_pdf_from_markdown(report_md, bibtex_dict)
                st.download_button("Download PDF", pdf_data, file_name="report.pdf", mime="application/pdf")


            except Exception as e:
                st.error(f"An error occurred: {e}")


# Run the app
if __name__ == "__main__":
    asyncio.run(main()) # Run the main function using asyncio.run