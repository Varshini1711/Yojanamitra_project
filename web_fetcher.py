import httpx
from bs4 import BeautifulSoup

async def fetch_scheme_page(url: str, max_chars: int = 3000) -> str:
    """Fetch a government scheme URL and return clean text."""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; SchemeBot/1.0)"}
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove clutter: scripts, styles, nav, footer
        for tag in soup(["script", "style", "nav",
                           "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        clean = " ".join(lines)

        return clean[:max_chars]  # Keep within token budget

    except Exception as e:
        print(f"Web fetch failed for {url}: {e}")
        return ""  # Return empty — fallback to static data