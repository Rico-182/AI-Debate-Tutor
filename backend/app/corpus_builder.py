"""
YouTube Debate Corpus Builder Pipeline

Extracts debate speeches from YouTube videos and builds corpus files.

Usage:
    python -m app.corpus_builder --urls "URL1" "URL2" "URL3" --output-dir ./corpus
    python3 corpus_builder.py --urls "https://youtu.be/j0Y4Yi6YQsk?si=YtxzFe1UwEnMSR6Z" "https://youtu.be/PIploGXuYAY?si=E0dB_5GmYzf0tkhs" --output-dir ./corpus
    python3 corpus_builder.py --urls "https://youtu.be/Q2bNhTAQspY?si=Qkn6pZTxT6kV_0kv" --output-dir ./corpus
    python3 corpus_builder.py --urls "https://youtu.be/kUDGECd7Wtw?si=gKO0-eV6HCZMvGGX" --output-dir ./corpus

Dependencies:
    pip install youtube-transcript-api openai python-dotenv
"""
import os
import re
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Try to import youtube transcript API
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    _HAS_YT_API = True
except ImportError:
    _HAS_YT_API = False
    print("[WARNING] youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None
        print("[WARNING] OpenAI client not available")


@dataclass
class ExtractedSpeech:
    motion_guess_1: str
    motion_guess_2: str
    side: str  # "Government" or "Opposition"
    speech_text: str
    raw_speech: str  # before cleaning


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(video_url: str) -> Optional[str]:
    """
    Fetch YouTube transcript using youtube-transcript-api.
    Returns raw transcript as a single string.
    """
    if not _HAS_YT_API:
        print("[ERROR] youtube-transcript-api not installed")
        return None

    video_id = extract_video_id(video_url)
    if not video_id:
        print(f"[ERROR] Could not extract video ID from {video_url}")
        return None

    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)  # returns a FetchedTranscript

        # Convert to raw dicts, then join text
        raw = fetched.to_raw_data()
        full_text = " ".join(entry["text"] for entry in raw)
        return full_text
    except Exception as e:
        print(f"[ERROR] Failed to get transcript for {video_id}: {e}")
        return None


def extract_speeches_and_motion(transcript: str, model: str = "gpt-4o-mini") -> Optional[Dict]:
    """
    Use LLM to extract first Government and Opposition speeches, and guess the motion.
    Returns JSON with extracted data.
    """
    if not client:
        print("[ERROR] OpenAI client not available")
        return None

    prompt = f"""You are analyzing a debate transcript. Extract the following:

1. The FIRST PROPOSITION/GOVERNMENT constructive speech (the opening speech for the proposition side)
2. The FIRST OPPOSITION constructive speech (the opening speech for the opposition side)
3. Make TWO educated guesses about what the motion being debated is

TRANSCRIPT:
{transcript}

Tasks:
- Remove filler words (um, uh, like when overused)
- Remove transitions like "Thank you [name] for that speech" or "I'd like to thank the previous speaker"
- Extract clean versions of the first two constructive speeches
- Keep the actual argumentation and content intact

Return STRICT JSON format:
{{
  "motion_guess_1": "This House believes that...",
  "motion_guess_2": "This House would...",
  "government_speech": "The cleaned first government/proposition speech text...",
  "opposition_speech": "The cleaned first opposition speech text..."
}}

If you cannot identify both speeches clearly, return null for that field.
Return ONLY valid JSON, no other text.
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a debate transcript analyzer. Always return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        result_text = resp.choices[0].message.content.strip()

        # Try to parse JSON - sometimes the model adds markdown formatting
        # Remove ```json and ``` if present
        result_text = re.sub(r'^```json\s*', '', result_text)
        result_text = re.sub(r'\s*```$', '', result_text)

        result = json.loads(result_text)
        return result
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON from LLM response: {e}")
        print(f"Response was: {result_text[:500]}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to extract speeches: {e}")
        return None


def inject_metadata_into_speech(
    speech_text: str,
    motion_1: str,
    motion_2: str,
    side: str,
    interval: int = 600
) -> str:
    """
    Inject metadata every ~interval words into the speech.

    Metadata format:
    [METADATA] Motion: {motion_1} OR {motion_2} | Side: {side}
    """
    words = speech_text.split()
    chunks = []

    metadata_line = f"\n\n[METADATA] Motion: {motion_1} OR {motion_2} | Side: {side}\n\n"

    i = 0
    while i < len(words):
        # Take next chunk of words
        chunk_words = words[i:i + interval]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        # Add metadata if not at the end
        if i + interval < len(words):
            chunks.append(metadata_line)

        i += interval

    # Add metadata at the beginning
    final_text = metadata_line + "".join(chunks)
    return final_text


def clean_and_format_speech(
    speech_text: str,
    motion_1: str,
    motion_2: str,
    side: str,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Final cleaning pass and format for .txt file, then inject metadata.
    """
    if not client:
        # If no client, just inject metadata without cleaning
        return inject_metadata_into_speech(speech_text, motion_1, motion_2, side)

    prompt = f"""Clean this debate speech for storage in a text corpus:

SPEECH:
{speech_text}

Tasks:
- Format for clean .txt storage
- Remove any remaining filler or artifacts
- Ensure proper paragraph breaks
- Keep all substantive argumentation
- Return ONLY the cleaned speech text, no extra commentary

Return the cleaned speech:
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a text cleaner. Return only the cleaned text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        cleaned = resp.choices[0].message.content.strip()

        # Inject metadata
        final_text = inject_metadata_into_speech(cleaned, motion_1, motion_2, side)
        return final_text
    except Exception as e:
        print(f"[ERROR] Failed to clean speech: {e}")
        # Fallback to just injecting metadata
        return inject_metadata_into_speech(speech_text, motion_1, motion_2, side)


def process_youtube_video(
    video_url: str,
    output_dir: str = "./corpus",
    model: str = "gpt-4o-mini"
) -> bool:
    """
    Full pipeline for one YouTube video:
    1. Get transcript
    2. Extract speeches and motion
    3. Clean and add metadata
    4. Save to files

    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {video_url}")
    print(f"{'='*60}")

    # Step 1: Get transcript
    print("[1/4] Fetching transcript...")
    transcript = get_youtube_transcript(video_url)
    if not transcript:
        return False
    print(f"  ✓ Transcript fetched ({len(transcript)} chars)")

    # Step 2: Extract speeches
    print("[2/4] Extracting speeches and motion...")
    extracted = extract_speeches_and_motion(transcript, model=model)
    if not extracted:
        return False

    motion_1 = extracted.get("motion_guess_1", "Unknown Motion")
    motion_2 = extracted.get("motion_guess_2", "Unknown Motion")
    gov_speech = extracted.get("government_speech")
    opp_speech = extracted.get("opposition_speech")

    print(f"  ✓ Motion guesses:")
    print(f"    1. {motion_1}")
    print(f"    2. {motion_2}")

    if not gov_speech or not opp_speech:
        print("[ERROR] Could not extract both speeches")
        return False

    # Step 3: Clean and format
    print("[3/4] Cleaning and adding metadata...")

    gov_final = clean_and_format_speech(gov_speech, motion_1, motion_2, "Government", model=model)
    opp_final = clean_and_format_speech(opp_speech, motion_1, motion_2, "Opposition", model=model)

    print(f"  ✓ Government speech: {len(gov_final)} chars")
    print(f"  ✓ Opposition speech: {len(opp_final)} chars")

    # Step 4: Save to files
    print("[4/4] Saving to corpus...")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Generate filenames based on motion
    video_id = extract_video_id(video_url) or "unknown"
    safe_motion = re.sub(r'[^a-zA-Z0-9\s]', '', motion_1)[:50].strip().replace(' ', '_')

    gov_filename = f"{video_id}_{safe_motion}_GOV.txt"
    opp_filename = f"{video_id}_{safe_motion}_OPP.txt"

    gov_path = os.path.join(output_dir, gov_filename)
    opp_path = os.path.join(output_dir, opp_filename)

    with open(gov_path, "w", encoding="utf-8") as f:
        f.write(gov_final)

    with open(opp_path, "w", encoding="utf-8") as f:
        f.write(opp_final)

    print(f"  ✓ Saved: {gov_filename}")
    print(f"  ✓ Saved: {opp_filename}")
    print(f"✅ Successfully processed {video_url}\n")

    return True


def process_youtube_batch(
    video_urls: List[str],
    output_dir: str = "./corpus",
    model: str = "gpt-4o-mini"
):
    """
    Process multiple YouTube videos in batch.
    """
    print(f"\n🎬 Processing {len(video_urls)} YouTube videos...")
    print(f"📁 Output directory: {output_dir}\n")

    successful = 0
    failed = 0

    for i, url in enumerate(video_urls, 1):
        print(f"\n[Video {i}/{len(video_urls)}]")
        success = process_youtube_video(url, output_dir, model)

        if success:
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"📊 BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Corpus location: {output_dir}")


# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build debate corpus from YouTube videos")
    parser.add_argument("--urls", nargs="+", required=True, help="YouTube video URL(s)")
    parser.add_argument("--output-dir", default="./corpus", help="Output directory for corpus files")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")

    args = parser.parse_args()

    # Check dependencies
    if not _HAS_YT_API:
        print("\n❌ Missing dependency: youtube-transcript-api")
        print("Install with: pip install youtube-transcript-api")
        exit(1)

    if not client:
        print("\n❌ OpenAI client not configured")
        print("Make sure OPENAI_API_KEY is set in your .env file")
        exit(1)

    # Process videos
    process_youtube_batch(args.urls, args.output_dir, args.model)
