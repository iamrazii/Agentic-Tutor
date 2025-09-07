from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from typing import TypedDict
import os,tempfile,re,subprocess,json
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)

from langchain.docstore.document import Document

def load_documents(file_objs):
        
    docs = []
    exe = os.path.join(os.path.dirname(__file__), "processor.exe")

    for f in file_objs:
        ext = os.path.splitext(f.name)[1].lower()
        if ext not in [".pdf", ".docx", ".txt"]:
            raise ValueError(f"Unsupported file type: {ext}")

        # Save UploadedFile to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name

        cmd = [exe, tmp_path]

        # Run in binary mode to avoid cp1252 decode issues
        result = subprocess.run(cmd, capture_output=True, text=False)

        stdout = result.stdout.decode("utf-8", errors="ignore")
        stderr = result.stderr.decode("utf-8", errors="ignore")

        if not stdout.strip():
            raise RuntimeError(f"Go processor returned no output. STDERR: {stderr}")

        go_docs = json.loads(stdout)
        for d in go_docs:
            docs.append(Document(
                page_content=d["page_content"],
                metadata=d["metadata"]
            ))

        os.remove(tmp_path)

    return docs
# ---------- Section Structure ----------
class Section(TypedDict):
    heading: str
    content: list[str]
    page: int


# ---------- Heading Detection ----------
def looks_like_reference(line: str) -> bool:
    line = line.strip()
    if not line:
        return False

    patterns = [
        re.compile(r"^\d+\s+[A-Z][a-z]"),
        re.compile(
            r"(Volume|Issue|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
            r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)",
            re.IGNORECASE,
        ),
        re.compile(r"\b(19|20)\d{2}\b"),
        re.compile(r'["""]'),
    ]

    if any(p.search(line) for p in patterns):
        return True

    if len(line.split()) > 12:
        return True

    return False


def detect_heading(text: str) -> str | None:
    lines = text.split("\n")
    heading_pattern = re.compile(r"^(?:\d+\.\s|#+\s|[A-Z ]{5,}|.*:)$")

    for line in lines:
        if looks_like_reference(line):
            continue
        line = line.strip()
        if not line:
            continue

        if (
            line.endswith(":")
            or line.istitle()
            or heading_pattern.match(line)
            or (line.isupper() and len(line.split()) <= 6)
            or line.startswith("#")
        ):
            return line

    return None


# ---------- Split into Sections ----------
def split_into_sections(docs) -> list[Section]:
    if not docs:
        return []
    
    sections: list[Section] = []
    current_section: Section | None = None
    
    for doc in docs:
        page_num = doc.metadata.get("page", -1)
        
        # Split and process lines in one go, filter empty lines
        lines = [line.strip() for line in doc.page_content.split("\n") if line.strip()]
        
        for line in lines:
            if detect_heading(line):
                # Finalize current section if it has content
                if current_section and current_section["content"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {"heading": line, "content": [], "page": page_num}
            else:
                # Initialize default section if needed
                if current_section is None:
                    current_section = {"heading": "General", "content": [], "page": page_num}
                
                current_section["content"].append(line)
    
    # Don't forget the last section
    if current_section and current_section["content"]:
        sections.append(current_section)
    
    return sections

# ---------- Chunk Sections ----------
def chunk_sections(sections, chunk_size=400, chunk_overlap=50):  # Reduced overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,  # Faster than token counting
        is_separator_regex=False
    )

    def process_section(sec):
        text = " ".join(sec["content"])
        if not text.strip() or len(text) < 50:  # Skip very short texts
            return []
        doc = Document(
            page_content=text,
            metadata={"heading": sec["heading"], "page": sec["page"]},
        )
        return splitter.split_documents([doc])

    all_chunks = []
    # Process larger sections first (more efficient)
    sorted_sections = sorted(sections, key=lambda x: len(" ".join(x["content"])), reverse=True)
    
    with ThreadPoolExecutor(max_workers=4) as ex:  # Increase workers
        results = ex.map(process_section, sorted_sections)
        for res in results:
            all_chunks.extend(res)

    return all_chunks


# ---------- Preprocessor ----------
class Preprocessor:
    def __init__(self, chunk_size=800, overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def load_and_process(self, objs):
        docs = load_documents(objs)
        print("Documents loaded\n")
        sections = split_into_sections(docs)
        chunks = chunk_sections(sections)
        print("chunks created\n")
        return chunks
    
    def build_retriever(self, chunks, k=25):
        """Optimized retriever builder with parallel processing"""
        # Filter out empty chunks upfront
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        
        # Build FAISS and BM25 in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            faiss_future = executor.submit(FAISS.from_documents, chunks, self.embeddings)
            bm25_future = executor.submit(BM25Retriever.from_documents, chunks)
            
            faiss_store = faiss_future.result()
            bm25 = bm25_future.result()
        
        bm25.k = k

        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_store.as_retriever(search_kwargs={"k": k}), bm25],
            weights=[0.5, 0.5],
        )

        strict_retriever = faiss_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.90, "k": 5},
        )

        return ensemble_retriever, strict_retriever

    def build_safe_retriever(self, ensemble_retriever, strict_retriever):
        """Streamlined SafeRetriever"""
        class SafeRetriever:
            def __init__(self, strict, ensemble):
                self.strict = strict
                self.ensemble = ensemble

            def invoke(self, query):
                strict_results = self.strict.invoke(query)
                if not strict_results:
                    return []
                return self.ensemble.invoke(query)

        return SafeRetriever(strict_retriever, ensemble_retriever)