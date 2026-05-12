#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import with proper error handling
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"Missing package: {e}")
    print("Please run: pip install langchain-community langchain-huggingface langchain-text-splitters")
    sys.exit(1)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import *
from graph import SelfRAGGraph

console = Console()

class KnowledgeBase:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        console.print("[bold yellow]Initializing embeddings...[/bold yellow]")
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                console.print(f"[green]✓ GPU detected: {torch.cuda.get_device_name(0)}[/green]")
            else:
                console.print("[dim]Using CPU for embeddings[/dim]")

            # Set environment variable to disable symlink warning on Windows
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            console.print("[green]✓ Embeddings initialized[/green]")
        except Exception as e:
            console.print(f"[red]Error initializing embeddings: {e}[/red]")
            raise

    def load_pdfs(self):
        documents = []
        pdf_files = list(Path(DATA_PATH).glob("*.pdf"))

        if not pdf_files:
            console.print(f"[red]No PDF files found in '{DATA_PATH}'[/red]")
            console.print("[yellow]Please add your 5 PDF files to the 'data' folder[/yellow]")
            return []

        console.print(f"[bold cyan]Loading {len(pdf_files)} PDF files...[/bold cyan]")

        for pdf_path in pdf_files:
            console.print(f"  Processing: {pdf_path.name}")
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = pdf_path.name

                    # Add department metadata
                    if "CS_" in pdf_path.name:
                        doc.metadata["department"] = "Computer Science"
                    elif "EE_" in pdf_path.name:
                        doc.metadata["department"] = "Electrical Engineering"
                    elif "BBA_" in pdf_path.name:
                        doc.metadata["department"] = "Business Administration"
                    elif "Policies" in pdf_path.name:
                        doc.metadata["department"] = "University"
                    elif "Faculty" in pdf_path.name:
                        doc.metadata["department"] = "University"
                    else:
                        doc.metadata["department"] = "General"

                documents.extend(docs)
                console.print(f"    ✓ Loaded {len(docs)} pages")
            except Exception as e:
                console.print(f"    [red]✗ Error: {e}[/red]")

        return documents

    def chunk_documents(self, documents):
        console.print("[bold yellow]Chunking documents...[/bold yellow]")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = text_splitter.split_documents(documents)
        console.print(f"[green]✓ Created {len(chunks)} chunks[/green]")
        return chunks

    def create_vector_store(self, chunks):
        console.print("[bold yellow]Creating vector store...[/bold yellow]")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=VECTOR_STORE_PATH
        )

        # Remove the .persist() call as it's deprecated
        console.print(f"[green]✓ Vector store created at '{VECTOR_STORE_PATH}'[/green]")
        return self.vector_store

    def load_vector_store(self):
        if os.path.exists(VECTOR_STORE_PATH):
            console.print("[bold yellow]Loading existing vector store...[/bold yellow]")
            try:
                self.vector_store = Chroma(
                    persist_directory=VECTOR_STORE_PATH,
                    embedding_function=self.embeddings
                )
                console.print("[green]✓ Vector store loaded[/green]")
                return True
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load vector store: {e}[/yellow]")
                return False
        return False

    def setup(self, force_rebuild=False):
        if not force_rebuild and self.load_vector_store():
            return self.vector_store

        documents = self.load_pdfs()
        if not documents:
            raise ValueError("No documents loaded. Please add PDF files to the 'data' folder.")

        chunks = self.chunk_documents(documents)
        return self.create_vector_store(chunks)

class SelfRAGAgent:
    def __init__(self, force_rebuild_kb=False):
        console.print(Panel.fit("[bold blue]Self-RAG Agent[/bold blue]", subtitle="University Course Advisory System"))

        self.kb = KnowledgeBase()
        try:
            self.vector_store = self.kb.setup(force_rebuild=force_rebuild_kb)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

        self.graph = SelfRAGGraph(self.vector_store)
        console.print("[green]✓ Agent ready![/green]\n")

    def run_query(self, query: str):
        console.print(f"\n[bold cyan]💬 Query:[/bold cyan] {query}")

        result = self.graph.run(query)

        # Get the final response
        final_response = result.get("final_response", "")
        current_response = result.get("current_response", "")

        # Use final_response if available, otherwise current_response
        response = final_response if final_response else current_response

        if not response or response.strip() == "":
            response = "I couldn't generate a response. Please try rephrasing your question."

        # Display response
        console.print(Panel(response, title="[bold green]Response[/bold green]", border_style="green"))

        # Show trace summary
        trace = result.get("execution_trace", [])
        if trace:
            table = Table(title="Execution Trace", show_header=True, header_style="bold")
            table.add_column("Step", style="cyan")
            table.add_column("Details", style="white")

            for entry in trace[-5:]:
                step = entry.get("step", "unknown")
                details = {k: v for k, v in entry.items() if k != "step"}
                details_str = str(details)[:60]
                table.add_row(step, details_str)

            console.print(table)

        return result

    def interactive_mode(self):
        console.print("[bold cyan]Interactive Mode[/bold cyan]")
        console.print("Ask questions about courses, policies, or faculty. Type 'quit' to exit.\n")

        while True:
            try:
                query = console.input("[bold yellow]You: [/bold yellow]").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("[bold]Goodbye![/bold]")
                    break

                if not query:
                    continue

                self.run_query(query)

            except KeyboardInterrupt:
                console.print("\n[bold]Goodbye![/bold]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Self-RAG Agent for University Course Advisory")
    parser.add_argument("--rebuild-kb", action="store_true", help="Force rebuild knowledge base")
    parser.add_argument("--query", type=str, help="Run single query")

    args = parser.parse_args()

    # Check for API key
    if not GROQ_API_KEY:
        console.print("[red]Error: GROQ_API_KEY not found in .env file[/red]")
        console.print("[yellow]Please create a .env file with: GROQ_API_KEY=your_key_here[/yellow]")
        console.print("[dim]Get your API key from: https://console.groq.com[/dim]")
        return

    agent = SelfRAGAgent(force_rebuild_kb=args.rebuild_kb)

    if args.query:
        agent.run_query(args.query)
    else:
        agent.interactive_mode()

if __name__ == "__main__":
    main()