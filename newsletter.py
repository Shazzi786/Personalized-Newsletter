# ============================================================
#  Personalized AI Newsletter - Main Script
#  Author: Your Name
#  Date: 2026-04-28
# ============================================================

# --- Standard Library ---
import os
import json
import time
import logging
import smtplib
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Environment & Configuration ---
from dotenv import load_dotenv          # pip: python-dotenv

# --- AI / LLM ---
import openai                           # pip: openai          (GPT-4 / GPT-3.5)
# import anthropic                      # pip: anthropic       (Claude – uncomment if using)
# import google.generativeai as genai  # pip: google-generativeai (Gemini – uncomment if using)

# --- News & Web Content Fetching ---
import requests                         # pip: requests
import feedparser                       # pip: feedparser      (RSS feeds)
from bs4 import BeautifulSoup           # pip: beautifulsoup4  (HTML scraping)
import newspaper                        # pip: newspaper4k     (article extraction)

# --- NLP & Summarization ---
from transformers import pipeline       # pip: transformers    (HuggingFace summarization)
import nltk                             # pip: nltk            (text processing)
# nltk.download('punkt')               # Run once to download NLTK data

# --- Data Handling ---
import pandas as pd                     # pip: pandas
import numpy as np                      # pip: numpy

# --- Scheduling ---
import schedule                         # pip: schedule        (run newsletter on a cron-like schedule)

# --- User Preference & Vector Search (for personalization) ---
from sentence_transformers import SentenceTransformer  # pip: sentence-transformers
import faiss                            # pip: faiss-cpu       (similarity search for topic matching)

# --- Database (user profiles & history) ---
import sqlite3                          # built-in
# from pymongo import MongoClient       # pip: pymongo         (uncomment for MongoDB)

# --- Template & HTML Rendering ---
from jinja2 import Environment, FileSystemLoader  # pip: jinja2

# ============================================================
#  Load environment variables from .env
# ============================================================
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
EMAIL_ADDRESS   = os.getenv("EMAIL_ADDRESS", "")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD", "")
SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", 587))

# ============================================================
#  Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("newsletter.log"),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
#  SECTION 1 – Fetch News Articles
# ============================================================

def fetch_rss_articles(feed_urls: list[str], max_per_feed: int = 5) -> list[dict]:
    """
    Fetch articles from a list of RSS feed URLs.

    Args:
        feed_urls:     List of RSS/Atom feed URLs.
        max_per_feed:  Maximum articles to collect per feed.

    Returns:
        List of article dicts with keys: title, link, summary, published.
    """
    articles = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_per_feed]:
            articles.append({
                "title":     entry.get("title", ""),
                "link":      entry.get("link", ""),
                "summary":   entry.get("summary", ""),
                "published": entry.get("published", ""),
                "source":    feed.feed.get("title", url),
            })
    logger.info(f"Fetched {len(articles)} articles from {len(feed_urls)} feeds.")
    return articles


def scrape_article_content(url: str) -> str:
    """
    Extract the full text of an article from its URL using newspaper4k.

    Args:
        url: URL of the article.

    Returns:
        Full article text as a string.
    """
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.warning(f"Could not scrape {url}: {e}")
        return ""


# ============================================================
#  SECTION 2 – Summarize & Personalize with AI
# ============================================================

def summarize_with_openai(text: str, topic: str = "", max_tokens: int = 200) -> str:
    """
    Summarize article text using OpenAI's Chat API.

    Args:
        text:       Full article text to summarize.
        topic:      User's interest topic for personalization context.
        max_tokens: Max tokens for the summary response.

    Returns:
        AI-generated summary string.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
        f"You are an expert newsletter writer. Summarize the following article "
        f"in 3-4 sentences, highlighting insights relevant to '{topic}'.\n\n"
        f"Article:\n{text[:3000]}"  # Truncate to avoid token limits
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def rank_articles_by_interest(articles: list[dict], user_interests: list[str]) -> list[dict]:
    """
    Rank articles by semantic similarity to the user's interests using
    SentenceTransformers + FAISS.

    Args:
        articles:        List of article dicts (must have 'title' and 'summary').
        user_interests:  List of topic strings the user cares about.

    Returns:
        Articles sorted by relevance score (highest first).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    interest_text   = " ".join(user_interests)
    interest_vec    = model.encode([interest_text])

    article_texts   = [a["title"] + " " + a.get("summary", "") for a in articles]
    article_vecs    = model.encode(article_texts)

    # Cosine similarity via FAISS
    index = faiss.IndexFlatIP(article_vecs.shape[1])
    faiss.normalize_L2(article_vecs)
    faiss.normalize_L2(interest_vec)
    index.add(article_vecs)

    distances, indices = index.search(interest_vec, len(articles))

    ranked = [articles[i] for i in indices[0]]
    for i, article in enumerate(ranked):
        article["relevance_score"] = float(distances[0][i])

    logger.info("Articles ranked by user interest.")
    return ranked


# ============================================================
#  SECTION 3 – Build the Newsletter HTML
# ============================================================

def build_newsletter_html(user_name: str, articles: list[dict], template_path: str = "templates") -> str:
    """
    Render the newsletter HTML using a Jinja2 template.

    Args:
        user_name:     Recipient's name for personalization.
        articles:      List of ranked & summarized article dicts.
        template_path: Directory containing Jinja2 templates.

    Returns:
        Rendered HTML string.
    """
    env      = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template("newsletter.html")

    html = template.render(
        user_name=user_name,
        articles=articles,
        date=datetime.date.today().strftime("%B %d, %Y"),
    )
    return html


# ============================================================
#  SECTION 4 – Send the Newsletter via Email
# ============================================================

def send_newsletter(recipient_email: str, subject: str, html_content: str) -> bool:
    """
    Send the newsletter HTML to a recipient via SMTP.

    Args:
        recipient_email: Destination email address.
        subject:         Email subject line.
        html_content:    Rendered HTML body.

    Returns:
        True if sent successfully, False otherwise.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_ADDRESS
    msg["To"]      = recipient_email
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
        logger.info(f"Newsletter sent to {recipient_email}.")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


# ============================================================
#  SECTION 5 – User Profile (SQLite)
# ============================================================

def init_db(db_path: str = "users.db") -> sqlite3.Connection:
    """
    Initialize the SQLite database for storing user profiles.

    Returns:
        SQLite connection object.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            email       TEXT    UNIQUE NOT NULL,
            interests   TEXT,           -- JSON list of interest strings
            frequency   TEXT    DEFAULT 'daily',
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def add_user(conn: sqlite3.Connection, name: str, email: str, interests: list[str], frequency: str = "daily"):
    """Add a new subscriber to the database."""
    conn.execute(
        "INSERT OR IGNORE INTO users (name, email, interests, frequency) VALUES (?, ?, ?, ?)",
        (name, email, json.dumps(interests), frequency),
    )
    conn.commit()
    logger.info(f"User '{name}' <{email}> added.")


def get_all_users(conn: sqlite3.Connection) -> list[dict]:
    """Retrieve all subscribers as a list of dicts."""
    cursor = conn.execute("SELECT name, email, interests, frequency FROM users")
    rows   = cursor.fetchall()
    return [
        {
            "name":      r[0],
            "email":     r[1],
            "interests": json.loads(r[2]) if r[2] else [],
            "frequency": r[3],
        }
        for r in rows
    ]


# ============================================================
#  SECTION 6 – Main Pipeline
# ============================================================

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://hnrss.org/frontpage",                          # Hacker News
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    # Add more feeds here...
]


def run_newsletter_pipeline():
    """
    Full end-to-end pipeline:
      1. Load users from DB
      2. Fetch latest articles from RSS feeds
      3. Rank & summarize per user
      4. Build HTML & send email
    """
    logger.info("=== Newsletter Pipeline Started ===")

    # 1. Load subscribers
    conn  = init_db()
    users = get_all_users(conn)

    if not users:
        logger.warning("No subscribers found. Add users with add_user() first.")
        return

    # 2. Fetch articles (shared across all users)
    articles = fetch_rss_articles(RSS_FEEDS, max_per_feed=10)

    for user in users:
        logger.info(f"Processing newsletter for {user['name']} …")

        # 3. Rank articles by personal interests
        ranked = rank_articles_by_interest(articles, user["interests"])
        top_articles = ranked[:5]  # Top 5 most relevant

        # 4. Summarize each article with AI
        for article in top_articles:
            content = scrape_article_content(article["link"])
            article["ai_summary"] = summarize_with_openai(
                content or article["summary"],
                topic=", ".join(user["interests"]),
            )

        # 5. Build HTML
        html = build_newsletter_html(
            user_name=user["name"],
            articles=top_articles,
        )

        # 6. Send
        subject = f"📰 Your Personalized AI Newsletter — {datetime.date.today().strftime('%B %d, %Y')}"
        send_newsletter(user["email"], subject, html)

    logger.info("=== Newsletter Pipeline Complete ===")


# ============================================================
#  Scheduler (optional – runs every day at 8 AM)
# ============================================================

def start_scheduler():
    """Start the daily newsletter scheduler."""
    schedule.every().day.at("08:00").do(run_newsletter_pipeline)
    logger.info("Scheduler started. Newsletter will send daily at 08:00.")
    while True:
        schedule.run_pending()
        time.sleep(60)


# ============================================================
#  Entry Point
# ============================================================

if __name__ == "__main__":
    # --- Quick test: add a sample user and run once ---
    conn = init_db()
    add_user(
        conn,
        name="Alice",
        email="alice@example.com",
        interests=["artificial intelligence", "machine learning", "startups"],
        frequency="daily",
    )

    # Run the pipeline once immediately
    run_newsletter_pipeline()

    # Uncomment to run on a daily schedule instead:
    # start_scheduler()
