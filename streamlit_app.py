import streamlit as st
import json
from ss2 import PaperEvaluator
import pandas as pd
import os
from dotenv import load_dotenv
import re
from io import StringIO

# Load environment variables
load_dotenv()


def extract_field(text, field):
    """Extract field value from text using regex with improved pattern"""
    pattern = f"{field}- (.*?)(?=\n[A-Z]{{2,}}- |\n\n|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        value = " ".join(
            " ".join(line.strip() for line in match.split("\n") if line.strip())
            for match in matches
        )
        return value
    return ""


def parse_pubmed_entry(entry):
    """Parse a single PubMed entry with improved field handling"""
    pmid_match = re.search(r"PMID-\s*(\d+)", entry)
    pmid = pmid_match.group(1) if pmid_match else ""

    # Title extraction
    title_pattern = r"TI\s*-\s*(.*?)(?=\n[A-Z]{2,}\s*-)"
    title_match = re.search(title_pattern, entry, re.DOTALL)
    title = (
        " ".join(line.strip() for line in title_match.group(1).split("\n"))
        if title_match
        else ""
    )

    # Abstract extraction
    abstract_pattern = r"AB\s*-\s*(.*?)(?=\n[A-Z]{2,}\s*-)"
    abstract_match = re.search(abstract_pattern, entry, re.DOTALL)
    abstract = (
        " ".join(line.strip() for line in abstract_match.group(1).split("\n"))
        if abstract_match
        else ""
    )

    # Publication date extraction
    pub_date = ""
    dp_match = re.search(r"DP\s*-\s*([^\n]+)", entry)
    if dp_match:
        date_text = dp_match.group(1).strip()
        year_match = re.search(r"\d{4}", date_text)
        month_match = re.search(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", date_text
        )
        if year_match and month_match:
            pub_date = f"{month_match.group(1)} {year_match.group()}"
        elif year_match:
            pub_date = year_match.group()

    data = {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "publication_date": pub_date,
    }

    return data


def parse_pubmed_file(content):
    """Parse PubMed file and return DataFrame"""
    # First, normalize line endings
    content = content.replace("\r\n", "\n")

    # Split entries more reliably
    entries = content.split("\n\nPMID- ")

    # Handle first entry which may not have the prefix
    if entries[0].startswith("PMID- "):
        first_entry = entries[0]
    else:
        first_entry = "PMID- " + entries[0]

    entries[0] = first_entry

    # Add PMID- prefix back to other entries
    entries[1:] = ["PMID- " + entry for entry in entries[1:]]

    parsed_entries = []
    # st.write(f"Found {len(entries)} entries to parse")  # Debug line

    for entry in entries:
        if entry.strip():
            parsed = parse_pubmed_entry(entry)
            if parsed["pmid"]:
                parsed_entries.append(parsed)

    df = pd.DataFrame(parsed_entries)

    # Clean up fields
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()

    # st.write(f"Successfully parsed {len(df)} entries")  # Debug line

    return df


def run_analysis(research_question, df, paper_limit):
    """Run the paper analysis with given parameters"""

    # Initialize evaluator
    openai_api_key = os.getenv("OPENAI_API_KEY")
    evaluator = PaperEvaluator(openai_api_key)

    # Create a placeholder for progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Limit number of papers based on user selection
        df = df.head(paper_limit)
        papers = df.to_dict("records")

        if not papers:
            st.warning("No papers found matching your criteria.")
            return None

        st.info(f"Found {len(papers)} papers to analyze")

        # Analyze papers
        results = []
        for idx, paper in enumerate(papers):
            title = paper["title"]
            abstract = paper.get("abstract", "No abstract available")
            pub_date = paper.get("publication_date")

            status_text.text(
                f"Analyzing paper {idx + 1} of {len(papers)}: {title[:50]}..."
            )
            progress_bar.progress((idx + 1) / len(papers))

            try:
                # Create a dictionary with all paper information
                paper_info = {
                    "title": title,
                    "publication_date": pub_date,
                    "abstract": abstract,
                    "pmid": paper.get("pmid", ""),
                }

                # Get AI analysis
                analysis = evaluator.evaluate_paper(
                    title=title,
                    abstract=abstract,
                    publication_date=pub_date,
                    research_question=research_question,
                )

                # Update the analysis with additional information
                analysis_dict = analysis.model_dump()
                analysis_dict.update(paper_info)

                # Create a new PaperAnalysis object with all information
                from ss2 import PaperAnalysis

                updated_analysis = PaperAnalysis(**analysis_dict)
                results.append(updated_analysis)

            except Exception as e:
                st.error(f"Error analyzing paper: {title}\nError: {str(e)}")

        # After all papers are analyzed, save relevant papers to candidates.json
        if results:
            relevant_papers = [
                paper.model_dump() for paper in results if paper.is_relevant
            ]

            if relevant_papers:
                with open("candidates.json", "w", encoding="utf-8") as f:
                    json.dump(relevant_papers, f, indent=2, ensure_ascii=False)
                # st.success(
                #     f"Saved {len(relevant_papers)} relevant papers to candidates.json"
                # )
            else:
                st.warning("No relevant papers found to save to candidates.json")

        return results
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None


def main():
    st.title("Research Paper Analysis Tool")

    # Create tabs at the start
    tab1, tab2, tab3 = st.tabs(["Summary View", "Raw JSON", "Saved Candidates"])

    # Move the candidates tab content outside the analysis block
    with tab3:
        try:
            if os.path.exists("candidates.json"):
                with open("candidates.json", "r", encoding="utf-8") as f:
                    candidates_data = json.load(f)

                st.markdown("### 📄 Saved Relevant Papers")
                st.markdown(f"Found {len(candidates_data)} saved papers")

                # Add download button for candidates.json
                st.download_button(
                    label="Download Candidates JSON",
                    data=json.dumps(candidates_data, indent=2),
                    file_name="candidates.json",
                    mime="application/json",
                )

                # Display the contents
                st.json(candidates_data)
            else:
                st.info(
                    "No saved candidates file found. Run an analysis to generate it."
                )
        except Exception as e:
            st.error(f"Error reading candidates file: {str(e)}")

    # Sidebar for inputs
    st.sidebar.header("Analysis Parameters")

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload PubMed text file", type=["txt"])

    # Initialize session state for DataFrame if not exists
    if "df" not in st.session_state:
        st.session_state.df = None

    # Process uploaded file
    if uploaded_file is not None:
        content = uploaded_file.getvalue().decode("utf-8")
        st.session_state.df = parse_pubmed_file(content)
        st.sidebar.success(f"Loaded {len(st.session_state.df)} papers from file")

        # Add paper limit slider after successful file upload
        paper_limit = st.sidebar.slider(
            "Number of papers to analyze",
            min_value=1,
            max_value=len(st.session_state.df),
            value=min(10, len(st.session_state.df)),  # Default to 10 or max available
        )
    else:
        paper_limit = 10  # Default value when no file is uploaded

    research_question = st.sidebar.text_area(
        "Research Question",
        "Does displacement of lower pole stones during retrograde intrarenal surgery improves stone-free status?",
    )

    # Run analysis button
    if st.sidebar.button("Run Analysis"):
        if st.session_state.df is None:
            st.error("Please upload a PubMed text file first")
        else:
            with st.spinner("Analyzing papers..."):
                results = run_analysis(
                    research_question, st.session_state.df, paper_limit
                )

                if results:
                    # Display results
                    st.success("Analysis completed!")

                    # Sort results to show relevant papers first
                    sorted_results = sorted(
                        results,
                        key=lambda x: (
                            not x.is_relevant,
                            -len(str(x.publication_date)) if x.publication_date else 0,
                        ),
                        reverse=False,
                    )

                    with tab1:
                        # Add summary statistics
                        relevant_count = sum(
                            1 for paper in sorted_results if paper.is_relevant
                        )

                        # Display summary with more emphasis
                        st.markdown("""
                        ### Summary
                        """)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Relevant Papers", relevant_count)
                        with col2:
                            st.metric("Total Papers", len(sorted_results))

                        # First show relevant papers
                        st.markdown("### 🟢 Relevant Papers")
                        for idx, paper in enumerate(sorted_results, 1):
                            if paper.is_relevant:
                                with st.expander(f"{idx}. {paper.title}"):
                                    st.markdown("""---""")
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(
                                            f"**Publication Date:** {paper.publication_date}"
                                        )
                                    with col2:
                                        if hasattr(paper, "pmid"):
                                            st.write(f"**PMID:** {paper.pmid}")

                                    # Display abstract in a quote block if available
                                    if hasattr(paper, "abstract") and paper.abstract:
                                        st.markdown("> **Abstract:**")
                                        st.markdown(f"> {paper.abstract}")
                                        st.markdown("---")

                                    st.write(f"**Study Type:** {paper.study_type}")
                                    st.write(f"**Sample Size:** {paper.sample_size}")

                                    # Handle stone-free rate that might be a dictionary
                                    if isinstance(paper.stone_free_rate, dict):
                                        st.write("**Stone-free Rates:**")
                                        for key, value in paper.stone_free_rate.items():
                                            st.write(f"- {key}: {value}")
                                    else:
                                        st.write(
                                            f"**Stone-free Rate:** {paper.stone_free_rate}"
                                        )

                                    st.write(
                                        f"**Methodology Quality:** {paper.methodology_quality}"
                                    )

                                    # Handle key findings that might be a dictionary
                                    if isinstance(paper.key_findings, dict):
                                        st.write("**Key Findings:**")
                                        for key, value in paper.key_findings.items():
                                            st.write(f"- {key}: {value}")
                                    else:
                                        st.write(
                                            "**Key Findings:**", paper.key_findings
                                        )

                                    st.write("**Limitations:**", paper.limitations)
                                    st.write(
                                        "**Relevance Explanation:**",
                                        paper.relevance_explanation,
                                    )

                        # Then show non-relevant papers
                        st.markdown("### ⚪ Other Papers")
                        for idx, paper in enumerate(sorted_results, 1):
                            if not paper.is_relevant:
                                with st.expander(f"{idx}. {paper.title}"):
                                    st.markdown("""---""")
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(
                                            f"**Publication Date:** {paper.publication_date}"
                                        )
                                    with col2:
                                        if hasattr(paper, "pmid"):
                                            st.write(f"**PMID:** {paper.pmid}")

                                    # Display abstract in a quote block if available
                                    if hasattr(paper, "abstract") and paper.abstract:
                                        st.markdown("> **Abstract:**")
                                        st.markdown(f"> {paper.abstract}")
                                        st.markdown("---")

                                    st.write(f"**Study Type:** {paper.study_type}")
                                    st.write(f"**Sample Size:** {paper.sample_size}")

                                    # Handle stone-free rate that might be a dictionary
                                    if isinstance(paper.stone_free_rate, dict):
                                        st.write("**Stone-free Rates:**")
                                        for key, value in paper.stone_free_rate.items():
                                            st.write(f"- {key}: {value}")
                                    else:
                                        st.write(
                                            f"**Stone-free Rate:** {paper.stone_free_rate}"
                                        )

                                    st.write(
                                        f"**Methodology Quality:** {paper.methodology_quality}"
                                    )

                                    # Handle key findings that might be a dictionary
                                    if isinstance(paper.key_findings, dict):
                                        st.write("**Key Findings:**")
                                        for key, value in paper.key_findings.items():
                                            st.write(f"- {key}: {value}")
                                    else:
                                        st.write(
                                            "**Key Findings:**", paper.key_findings
                                        )

                                    st.write("**Limitations:**", paper.limitations)
                                    st.write(
                                        "**Relevance Explanation:**",
                                        paper.relevance_explanation,
                                    )

                    with tab2:
                        # Convert results to JSON for display
                        json_results = [paper.model_dump() for paper in results]
                        st.json(json_results)

                        # Add download button
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(json_results, indent=2),
                            file_name="meta_analysis_data.json",
                            mime="application/json",
                        )


if __name__ == "__main__":
    main()
