"""
Process Banco_de_soluciones.csv:
1. Decode HTML in content_rich (strip tags + decode entities)
2. Translate en_US/pt_BR/pt_PT rows to Spanish using deep-translator
3. Save to Banco_de_soluciones_processed.csv
"""

import time
import pandas as pd
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

INPUT_PATH = "data/Banco_de_soluciones.csv"
OUTPUT_PATH = "data/Banco_de_soluciones_processed.csv"
FIELDS_TO_TRANSLATE = ["title", "description", "content_text", "content_rich"]
LANG_MAP = {
    "en_US": "en",
    "pt_BR": "pt",
    "pt_PT": "pt",
}


def clean_html(html_text: str) -> str:
    """Strip HTML tags and decode entities, returning plain text."""
    if not isinstance(html_text, str) or not html_text.strip():
        return html_text
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ")
    # Collapse multiple whitespace/newlines
    return " ".join(text.split())


def translate_text(text: str, source_lang: str) -> str:
    """Translate text to Spanish using GoogleTranslator."""
    if not isinstance(text, str) or not text.strip():
        return text
    # GoogleTranslator has a 5000 char limit per request
    max_chunk = 4900
    if len(text) <= max_chunk:
        return GoogleTranslator(source=source_lang, target="es").translate(text)
    # Split long texts into chunks at sentence boundaries
    chunks = []
    current = ""
    for sentence in text.replace(". ", ".|").split("|"):
        if len(current) + len(sentence) > max_chunk:
            if current:
                chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        chunks.append(current)
    translated_chunks = []
    for chunk in chunks:
        translated_chunks.append(
            GoogleTranslator(source=source_lang, target="es").translate(chunk)
        )
        time.sleep(0.3)
    return " ".join(translated_chunks)


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows")
    print(f"Language distribution:\n{df['lang'].value_counts()}\n")

    # Track original language
    df["original_lang"] = df["lang"]

    # Step 1: Clean HTML in content_rich for ALL rows
    print("Cleaning HTML from content_rich...")
    df["content_rich"] = df["content_rich"].apply(clean_html)
    print("Done cleaning HTML.\n")

    # Step 2: Translate non-Spanish rows
    non_es = df[df["lang"] != "es_PY"]
    print(f"Rows to translate: {len(non_es)}")

    for idx, row in non_es.iterrows():
        lang_code = LANG_MAP.get(row["lang"])
        if not lang_code:
            print(f"  Skipping unknown lang: {row['lang']} at row {idx}")
            continue

        if idx % 50 == 0:
            print(f"  Translating row {idx} (lang={row['lang']})...")

        for field in FIELDS_TO_TRANSLATE:
            original = row[field]
            if isinstance(original, str) and original.strip():
                try:
                    df.at[idx, field] = translate_text(original, lang_code)
                    time.sleep(0.2)
                except Exception as e:
                    print(f"  Error translating {field} at row {idx}: {e}")

        df.at[idx, "lang"] = "es_PY"

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")

    # Verification
    df_out = pd.read_csv(OUTPUT_PATH)
    print(f"\nVerification:")
    print(f"  Row count: {len(df_out)}")
    print(f"  Language distribution:\n{df_out['lang'].value_counts()}")
    print(f"  Original language distribution:\n{df_out['original_lang'].value_counts()}")
    html_remaining = df_out["content_rich"].str.contains("<[a-zA-Z]", regex=True, na=False).sum()
    print(f"  Rows with remaining HTML tags in content_rich: {html_remaining}")


if __name__ == "__main__":
    main()
