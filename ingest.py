"""
PDF Ingestion Pipeline

Run once or when new papers are added.

Pipeline per PDF:
  1. Extract plain text per page (PyMuPDF)
  2. Detect and crop images/tables per page (PyMuPDF bbox detection)
  3. Send each cropped image to Nvidia Nemotron VLM (via OpenRouter) -> Markdown
  4. Stitch page text + VLM markdown into one chunk per page
  5. Embed the chunk via Nvidia NIM (nvidia/nv-embed-v2)
  6. Insert into local Qdrant collection

"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from langsmith.wrappers import wrap_openai

load_dotenv()

# Sanitize LangSmith environment variables to prevent UnicodeEncodeError
for env_key in ["LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"]:
    if val := os.getenv(env_key):
        sanitized = val.replace("—", "-").replace("–", "-")
        sanitized = "".join(c for c in sanitized if ord(c) < 128).strip()
        os.environ[env_key] = sanitized

console = Console()

# Config
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./local_research_db")
COLLECTION = os.getenv("DB_COLLECTION", "research_papers")
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", "./research_papers"))
# Vision model
VLM_MODEL = os.getenv("VLM_MODEL", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "baai/bge-m3")
EMBED_DIM = 1024  # baai/bge-m3 output dimension

# API clients
# _nvidia_vlm: wrapped with LangSmith so VLM chat completions are traced
_nvidia_vlm = wrap_openai(
    AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )
)

# _nvidia_embed: NOT wrapped -> langsmith.wrap_openai breaks the /embeddings endpoint
_nvidia_embed = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)


# Helpers
def _pil_to_base64(img: Image.Image) -> str:
    """
    Convert a PIL image to a base64 encoded PNG string.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def _vlm_image_to_markdown(img_b64: str) -> str:
    """
    Convert a cropped page image to Markdown using Nvidia NIM vision model.
    Uses nvidia/llama-3.1-nemotron-nano-vl-8b-v1
    """
    try:
        response = await _nvidia_vlm.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a cropped image of a table or figure from an academic paper. "
                                "Convert it to clean, precise Markdown. "
                                "For tables: use proper Markdown table syntax. "
                                "For figures/charts: describe the key data points and trends concisely. "
                                "Return only the Markdown, no commentary."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        console.print(
            f"[yellow]  \u26a0 VLM call failed ({exc}), skipping image[/yellow]"
        )
        return ""


async def _embed_text(text: str) -> list[float]:
    """
    Embed text using baai/bge-m3 via Nvidia NIM (1024 dims).
    Uses _nvidia_embed (unwrapped) —> wrap_openai breaks the /embeddings path.
    """
    response = await _nvidia_embed.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        encoding_format="float",
    )
    return response.data[0].embedding


def _extract_visual_crops(page: fitz.Page) -> list[Image.Image]:
    """
    Find image/table regions on a page and return them as PIL images.
    Strategy: use PyMuPDF image list + text block analysis to find
    non text bounding boxes, then render each crop at 2× resolution.
    """
    crops: list[Image.Image] = []
    mat = fitz.Matrix(2, 2)  # 2× scale for better VLM accuracy

    # Get image bboxes embedded directly in the page
    for img_info in page.get_image_info(xrefs=True):
        bbox = fitz.Rect(img_info["bbox"])
        # Skip tiny images (likely decorative icons)
        if bbox.width < 50 or bbox.height < 50:
            continue
        pix = page.get_pixmap(matrix=mat, clip=bbox)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        crops.append(img)

    return crops


async def _process_page(
    page: fitz.Page,
    page_num: int,
    filename: str,
) -> tuple[str, list[float]]:
    """
    Process one PDF page:
    1. Extract plain text
    2. Crop images/tables -> VLM -> Markdown
    3. Stitch together
    4. Embed the stitched chunk
    Returns (combined_text, embedding_vector)
    """
    plain_text = page.get_text("text").strip()

    # Find and convert visual elements
    visual_crops = _extract_visual_crops(page)
    vlm_sections: list[str] = []
    for crop_img in visual_crops:
        b64 = _pil_to_base64(crop_img)
        md = await _vlm_image_to_markdown(b64)
        if md.strip():
            vlm_sections.append(md.strip())

    # Stitch plain text + VLM markdown
    parts = [plain_text] if plain_text else []
    parts.extend(vlm_sections)
    combined = "\n\n".join(parts)

    if not combined.strip():
        return "", []

    vector = await _embed_text(combined)
    return combined, vector


async def ingest_pdf(pdf_path: Path, qdrant: AsyncQdrantClient) -> int:
    """
    Ingest a single PDF. Returns number of pages processed.
    """
    doc = fitz.open(str(pdf_path))
    filename = pdf_path.name
    points: list[PointStruct] = []

    console.print(f"\n[bold cyan]📄 {filename}[/bold cyan] ({len(doc)} pages)")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Ingesting pages...", total=len(doc))

        for page_num in range(len(doc)):
            page = doc[page_num]
            progress.update(task, description=f"Page {page_num + 1}/{len(doc)}")

            text, vector = await _process_page(page, page_num + 1, filename)
            if not text or not vector:
                progress.advance(task)
                continue

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": text,
                    "filename": filename,
                    "page": page_num + 1,
                    "source": str(pdf_path),
                },
            )
            points.append(point)
            progress.advance(task)

    if points:
        await qdrant.upsert(collection_name=COLLECTION, points=points)

    doc.close()
    return len(points)


async def main() -> None:
    console.print(
        Panel(
            "[bold green]Stateful PDF Chat Ingestion Pipeline[/bold green]\n"
            f"Papers directory: [cyan]{PAPERS_DIR}[/cyan]\n"
            f"Qdrant:           [cyan]{QDRANT_PATH}/{COLLECTION}[/cyan]\n"
            f"VLM model:        [cyan]{VLM_MODEL}[/cyan]\n"
            f"Embed model:      [cyan]{EMBED_MODEL}[/cyan]",
            title="Ingestion",
        )
    )

    # Find PDFs
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        console.print(
            f"[yellow]No PDF files found in {PAPERS_DIR}. "
            "Add papers and re run.[/yellow]"
        )
        return

    console.print(f"\nFound [bold]{len(pdf_files)}[/bold] PDF file(s):")
    for f in pdf_files:
        console.print(f"  • {f.name}")

    # Initialise Qdrant collection
    qdrant = AsyncQdrantClient(path=QDRANT_PATH)
    existing = [c.name for c in (await qdrant.get_collections()).collections]
    if COLLECTION not in existing:
        await qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        console.print(f"\n[green]Created Qdrant collection:[/green] {COLLECTION}")
    else:
        console.print(
            f"\n[yellow]Collection '{COLLECTION}' already exists. Upserting.[/yellow]"
        )

    total_pages = 0
    for pdf_path in pdf_files:
        pages = await ingest_pdf(pdf_path, qdrant)
        total_pages += pages
        console.print(f"  [green] [/green] {pdf_path.name} -> {pages} chunks indexed")

    await qdrant.close()

    console.print(
        Panel(
            f"[bold green]Done![/bold green]\n"
            f"Indexed [bold]{total_pages}[/bold] page chunks across [bold]{len(pdf_files)}[/bold] PDF(s).\n"
            f"LangSmith traces: [cyan]https://smith.langchain.com[/cyan]\n\n"
            f"Next steps:\n"
            f"  [cyan]python worker.py[/cyan]          ← start Temporal worker\n"
            f"  [cyan]uvicorn server:app --reload[/cyan] ← start FastAPI server",
            title="Ingestion Complete",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
