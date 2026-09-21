"""
arXiv Paper Ingestion DAG
==========================

Daily pipeline that fetches CS.AI papers from arXiv, processes PDFs,
indexes them with embeddings into OpenSearch, and stores results in PostgreSQL.

Schedule: Monday-Friday at 6 AM UTC
Pipeline: setup -> fetch -> index -> verify -> report -> cleanup
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Import task functions from modular structure
from arxiv_ingestion.setup import setup_environment
from arxiv_ingestion.fetching import fetch_daily_papers
from arxiv_ingestion.indexing import index_papers_hybrid, verify_hybrid_index
from arxiv_ingestion.reporting import generate_daily_report

# Default DAG arguments
default_args = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 8),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=30),
    "catchup": False,
}

# Create the DAG
dag = DAG(
    "arxiv_paper_ingestion",
    default_args=default_args,
    description="Daily arXiv CS.AI paper pipeline: fetch, parse PDFs, chunk, embed, and index for hybrid search",
    schedule="0 6 * * 1-5",  # Monday-Friday at 6 AM UTC
    max_active_runs=1,
    catchup=False,
    tags=["arxiv", "papers", "ingestion", "hybrid-search"],
)

# Task 1: Verify all services are healthy
setup_task = PythonOperator(
    task_id="setup_environment",
    python_callable=setup_environment,
    dag=dag,
)

# Task 2: Fetch papers, download PDFs, parse, store to DB
fetch_task = PythonOperator(
    task_id="fetch_daily_papers",
    python_callable=fetch_daily_papers,
    dag=dag,
)

# Task 3: Chunk papers, generate embeddings, index to OpenSearch (Week 3)
index_task = PythonOperator(
    task_id="index_papers_hybrid",
    python_callable=index_papers_hybrid,
    dag=dag,
)

# Task 4: Verify hybrid index health (Week 3)
verify_task = PythonOperator(
    task_id="verify_hybrid_index",
    python_callable=verify_hybrid_index,
    dag=dag,
)

# Task 5: Generate daily summary report
report_task = PythonOperator(
    task_id="generate_daily_report",
    python_callable=generate_daily_report,
    dag=dag,
)

# Task 6: Clean up temporary files
cleanup_task = BashOperator(
    task_id="cleanup_temp_files",
    bash_command="""
    echo "Cleaning up temporary files..."
    # Remove PDFs older than 30 days to manage disk space
    find /tmp -name "*.pdf" -type f -mtime +30 -delete 2>/dev/null || true
    echo "Cleanup completed"
    """,
    dag=dag,
)

# Task dependencies — full pipeline with indexing
setup_task >> fetch_task >> index_task >> verify_task >> report_task >> cleanup_task
