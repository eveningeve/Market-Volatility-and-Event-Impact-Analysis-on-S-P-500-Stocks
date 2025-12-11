# Market Volatility and Event Impact Analysis on S&P 500 Stocks

## Project Overview
This project performs an in-depth quantitative analysis of extreme one-day price drops in S&P 500 stocks (1970–2024, focus on 2019–2024). The core objective is to move beyond simply measuring volatility by **identifying and analyzing firms that consistently survive and thrive** across multiple structural crisis regimes (Dotcom, GFC, COVID-19).

We utilize a dual-threshold event study framework to segment crisis events and develop a **multi-crisis resilience profile**. This profile is then validated using both quantitative event-study layers (CAR(0,5)) and a qualitative RAG (Retrieval-Augmented Generation) agent to link resilience to fundamental business strategy.

## Key Findings

1.  **Structural Regime Shift:** The market exhibits a **structurally elevated volatility regime** post-2021, with the frequency of extreme events approximately **1.7x higher** than the pre-COVID baseline.
2.  **Idiosyncratic Risk Dominance:** The primary source of short-term tail risk is **idiosyncratic (firm-specific) crashes**, which are statistically more severe than systematic ones and produce the highest incidence of $\mathbf{7-10\sigma}$ collapses.
3.  **Identifying Crisis Thrivers (Tier 1):** We successfully stratified the S\&P 500 into resilience tiers, identifying a select group of **'Crisis Thrivers' (Tier 1)** that demonstrated the fastest and most complete price recovery across all three historical crises.
4.  **Short-Run Outperformance:** The **CAR(0,5) model** confirms that Tier 1 firms are not merely long-term survivors; they exhibit **immediate short-run outperformance** following a shock—losing less in crashes and capturing the strongest gains in the immediate relief rallies (e.g., $\mathbf{\sim+20\%}$ CAR in the Lehman rebound).
5.  **Qualitative Validation:** A specialized **RAG agent** analyzing 10-K MD\&A sections successfully linked the quantitative Tier 1 status to common qualitative factors, such as **disciplined capital allocation** and **operational flexibility**.

## Methodology Highlights

| Model / Layer | Purpose | Core Mechanism |
| :--- | :--- | :--- |
| **Event Classification** | Define and segment extreme price drops. | Dual-Threshold: $\ge 5\%$ drop AND $\le -2.5\sigma$ Z-score. Systematic crashes require simultaneous $\ge 2.0\%$ S&P 500 drop. |
| **Resilience Tiers** | Stratify firms based on multi-crisis recovery ability. | Composite score using Maximum Drawdown, Recovery Time, and Recovery Slope across Dotcom, GFC, and COVID-19. |
| **CAR(0,5) Model** | Quantify short-run P&L differences between tiers. | Compares the Cumulative Abnormal Return (CAR) over the 6-day window ($\mathbf{t=0}$ to $\mathbf{t=5}$) following a major negative shock. |
| **RAG Agent** | Qualitatively validate tiers using management strategy. | **Vector Search (Pinecone)** over 10-K MD\&A sections; **LLM Synthesis (GPT-4o-mini)** to generate insights on success factors. |

## Technology Stack

| Category | Tools Used | Purpose |
| :--- | :--- | :--- |
| **Data Processing** | Python, PySpark, Pandas | Distributed data processing (CRSP), transformation, and analysis. |
| **Vector Database** | Pinecone | Storage and retrieval of vectorized 10-K MD\&A text chunks. |
| **Generative AI** | OpenAI (GPT-4o-mini), Sentence Transformers | LLM synthesis for insight generation; embedding model for vector search. |
| **Deployment** | Streamlit, Streamlit Cloud | Creating an interactive, deployable web interface for the RAG agent. |
| **Visualization** | Matplotlib, JupyterLab | Interactive development and static visualization of results. |

## Getting Started (Local Setup)

To run the RAG agent locally:

1.  **Clone the Repository:**
    ```bash
    git clone [your-repository-url]
    cd [your-project-folder]
    ```

2.  **Set Up Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure Secrets:**
    Create a file named **`.env`** in the root directory and populate it with your API keys (this file is ignored by Git).
    ```
    # .env
    PINECONE_API_KEY="YOUR_KEY_HERE"
    OPENAI_API_KEY="YOUR_KEY_HERE"
    ```

4.  **Run the RAG Agent (Command Line):**
    ```bash
    python query_rag.py --query "What were the key capital allocation strategies used by Tier 1 Thrivers?"
    ```

5.  **Run the Streamlit App (Web Interface):**
    ```bash
    streamlit run streamlit_app.py
    ```
---

## Generative AI Disclaimer

Elements of this research workflow were supported through the use of Generative Artificial Intelligence (GenAI) tools. GenAI was used to provide guidance on syntax selection, package options, and debugging assistance for coding tasks related to data processing and analysis. All final code, results, and interpretations were independently reviewed and validated by the research team.

## Other Contributors
Junhan Chen, Nicholas Howard, Mohammed Zaid Bin Haris
