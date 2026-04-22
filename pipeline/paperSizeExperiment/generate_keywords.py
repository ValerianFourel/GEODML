#!/usr/bin/env python3
"""Generate 1000+ diverse search keywords using LLM via HuggingFace API.

Generates keywords across many verticals and topics, then deduplicates
and saves to keywords.txt.

Usage:
  python paperSizeExperiment/generate_keywords.py
  python paperSizeExperiment/generate_keywords.py --target 1000
  python paperSizeExperiment/generate_keywords.py --target 500 --model "Qwen/Qwen2.5-72B-Instruct"
"""

import argparse
import os
import re
import sys
import time
import random
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(PROJECT_ROOT / ".env.local")

HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Topic tree: broad categories -> subcategories ─────────────────────────────
# Each subcategory gets a targeted prompt to the LLM asking for 20-30 keywords.

TOPIC_TREE = {
    "B2B SaaS & Enterprise Software": [
        "CRM and sales tools",
        "HR and people management",
        "accounting and finance",
        "project and task management",
        "marketing automation and analytics",
        "customer support and helpdesk",
        "ERP and operations",
        "communication and collaboration",
        "cybersecurity and compliance",
        "developer tools and DevOps",
    ],
    "E-commerce & Retail": [
        "online store platforms",
        "payment processing",
        "shipping and logistics",
        "product sourcing and dropshipping",
        "retail POS systems",
        "loyalty programs and rewards",
        "inventory and supply chain",
        "pricing and repricing tools",
    ],
    "Healthcare & Medical": [
        "telemedicine and virtual care",
        "medical devices and equipment",
        "health insurance and billing",
        "mental health and therapy",
        "fitness and wellness apps",
        "pharmacy and prescriptions",
        "clinical trials and research",
        "patient management systems",
    ],
    "Education & E-learning": [
        "online course platforms",
        "tutoring and test prep",
        "language learning",
        "K-12 education technology",
        "university and higher education tools",
        "corporate training and LMS",
        "educational games and apps",
        "student information systems",
    ],
    "Finance & Fintech": [
        "personal banking and neobanks",
        "investment and trading platforms",
        "cryptocurrency and blockchain",
        "insurance comparison and insurtech",
        "tax preparation and filing",
        "budgeting and personal finance",
        "lending and mortgage",
        "wealth management and robo-advisors",
    ],
    "Real Estate & Property": [
        "property listing and search",
        "property management software",
        "mortgage and home loans",
        "real estate CRM",
        "commercial real estate",
        "home inspection and appraisal",
        "interior design and renovation",
        "smart home and IoT",
    ],
    "Travel & Hospitality": [
        "hotel and accommodation booking",
        "flight and airline search",
        "car rental and transportation",
        "travel planning and itineraries",
        "restaurant reservation and food delivery",
        "tour and activity booking",
        "travel insurance",
        "vacation rental management",
    ],
    "Legal & Compliance": [
        "legal practice management",
        "contract and document automation",
        "compliance and regulatory",
        "intellectual property and patents",
        "court and case management",
        "e-discovery and litigation",
        "immigration and visa services",
        "legal research tools",
    ],
    "Marketing & Advertising": [
        "SEO tools and rank tracking",
        "social media management",
        "content marketing and blogging",
        "PPC and paid advertising",
        "influencer marketing",
        "email marketing platforms",
        "video marketing and creation",
        "affiliate marketing and tracking",
    ],
    "Manufacturing & Industrial": [
        "CAD and product design",
        "quality management and inspection",
        "production planning and MES",
        "industrial automation and robotics",
        "3D printing and additive manufacturing",
        "environmental and safety management",
        "plant maintenance and CMMS",
        "supply chain and procurement",
    ],
    "Food & Agriculture": [
        "farm management software",
        "food safety and traceability",
        "restaurant management",
        "meal planning and nutrition",
        "grocery delivery and shopping",
        "agricultural drones and precision farming",
        "food manufacturing and processing",
        "wine and beverage industry",
    ],
    "Media & Entertainment": [
        "video streaming platforms",
        "music production and distribution",
        "podcast hosting and creation",
        "gaming platforms and tools",
        "photo editing and design",
        "news and publishing platforms",
        "event ticketing and management",
        "animation and VFX",
    ],
    "Automotive & Transportation": [
        "car buying and selling",
        "fleet management and telematics",
        "electric vehicles and charging",
        "auto repair and maintenance",
        "ride-sharing and mobility",
        "trucking and freight",
        "vehicle tracking and GPS",
        "auto insurance and claims",
    ],
    "Energy & Utilities": [
        "solar and renewable energy",
        "energy management and monitoring",
        "oil and gas technology",
        "utility billing and metering",
        "carbon tracking and sustainability",
        "electric grid and smart grid",
        "battery storage and systems",
        "energy trading platforms",
    ],
    "Nonprofit & Government": [
        "fundraising and donor management",
        "grant management",
        "volunteer management",
        "government services and e-gov",
        "civic engagement and voting",
        "disaster and emergency management",
        "public health and epidemiology",
        "environmental monitoring",
    ],
    "Construction & Architecture": [
        "construction project management",
        "building information modeling BIM",
        "estimating and takeoff software",
        "field service and inspection",
        "architecture and design tools",
        "contractor management",
        "permit and compliance management",
        "heavy equipment management",
    ],
    "AI & Data Science": [
        "machine learning platforms",
        "data labeling and annotation",
        "business intelligence dashboards",
        "data warehouse and ETL tools",
        "natural language processing tools",
        "computer vision software",
        "AI chatbot builders",
        "predictive analytics platforms",
    ],
    "Personal Services & Lifestyle": [
        "dating apps and matchmaking",
        "pet care and veterinary",
        "home cleaning and services",
        "personal styling and fashion",
        "moving and relocation services",
        "wedding planning and vendors",
        "beauty and salon booking",
        "childcare and babysitting",
    ],
    "Telecommunications": [
        "VoIP and business phone",
        "mobile device management",
        "network monitoring and management",
        "unified communications",
        "internet service providers",
        "5G infrastructure and services",
        "satellite and connectivity",
        "call center and contact center software",
    ],
    "Sports & Recreation": [
        "sports team management",
        "fitness tracking and wearables",
        "sports betting and fantasy",
        "gym and studio management",
        "outdoor recreation and camping",
        "sports coaching and training",
        "esports and competitive gaming",
        "swim school and youth sports",
    ],
}


def generate_keywords_for_topic(client, model_id, category, subcategory, existing_keywords):
    """Ask LLM to generate 25 search keywords for a specific subcategory."""
    # Sample some existing keywords to show format
    examples = random.sample(existing_keywords, min(5, len(existing_keywords))) if existing_keywords else []
    examples_text = "\n".join(f"  - {kw}" for kw in examples) if examples else "  - CRM software\n  - best project management tools"

    prompt = f"""Generate exactly 25 unique search keywords that a real person would type into Google.

Category: {category}
Subcategory: {subcategory}

Requirements:
- Keywords should be what real users actually search for (commercial, informational, or comparison queries)
- Mix of formats: "[thing] software", "best [thing]", "[thing] for [audience]", "how to [action]", "[thing] vs [thing]", "[thing] pricing", "[thing] alternatives"
- Each keyword should be 2-6 words long
- No duplicates, no numbering, no explanations
- Cover diverse specific topics within this subcategory
- Do NOT repeat any of these existing keywords:
{examples_text}

Return ONLY the keywords, one per line. No numbering, no bullets, no extra text."""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model_id, max_tokens=1000, temperature=0.8,
        )
        raw = response.choices[0].message.content.strip()
        # Strip <think> tags (DeepSeek)
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        keywords = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering, bullets
            line = re.sub(r'^[\d]+[.)]\s*', '', line)
            line = re.sub(r'^[-*•]\s*', '', line)
            line = line.strip().strip('"').strip("'")
            if line and 2 <= len(line.split()) <= 8 and len(line) < 80:
                keywords.append(line.lower())
        return keywords
    except Exception as e:
        print(f"    Error: {str(e)[:80]}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate diverse keywords via LLM")
    parser.add_argument("--target", type=int, default=1000,
                        help="Target number of unique keywords (default: 1000)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--output", type=str, default=str(SCRIPT_DIR / "keywords.txt"),
                        help="Output keywords file")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing keywords instead of overwriting")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("HF_TOKEN not set in .env.local")
        sys.exit(1)

    client = InferenceClient(token=HF_TOKEN)
    output_path = Path(args.output)

    # Load existing keywords if appending
    existing = set()
    if args.append and output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    existing.add(line.lower())
        print(f"Loaded {len(existing)} existing keywords")

    all_keywords = set(existing)

    # Count total subcategories
    total_subcats = sum(len(subs) for subs in TOPIC_TREE.values())
    print(f"\nTopic tree: {len(TOPIC_TREE)} categories, {total_subcats} subcategories")
    print(f"Target: {args.target} unique keywords")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}\n")

    subcat_num = 0
    for category, subcategories in TOPIC_TREE.items():
        print(f"\n{'─'*60}")
        print(f"  {category}")
        print(f"{'─'*60}")

        for subcategory in subcategories:
            subcat_num += 1

            if len(all_keywords) >= args.target:
                print(f"  [{subcat_num}/{total_subcats}] Target reached ({len(all_keywords)} keywords). Stopping.")
                break

            print(f"  [{subcat_num}/{total_subcats}] {subcategory}...", end=" ", flush=True)

            new_kws = generate_keywords_for_topic(
                client, args.model, category, subcategory, list(all_keywords)
            )

            # Deduplicate
            added = 0
            for kw in new_kws:
                if kw not in all_keywords:
                    all_keywords.add(kw)
                    added += 1

            print(f"got {len(new_kws)}, added {added} new (total: {len(all_keywords)})")
            time.sleep(random.uniform(0.5, 1.5))

        if len(all_keywords) >= args.target:
            break

    # If we still need more, do a second pass with more creative prompts
    if len(all_keywords) < args.target:
        print(f"\n\nSecond pass — need {args.target - len(all_keywords)} more keywords...")
        extra_topics = [
            "niche B2B tools for specific industries",
            "consumer product comparison queries",
            "how-to and tutorial searches",
            "location-based service searches",
            "pricing and cost comparison queries",
            "review and rating searches",
            "alternative and replacement searches",
            "integration and compatibility queries",
            "certification and training searches",
            "open source software searches",
        ]
        for topic in extra_topics:
            if len(all_keywords) >= args.target:
                break
            print(f"  Extra: {topic}...", end=" ", flush=True)
            new_kws = generate_keywords_for_topic(
                client, args.model, "Miscellaneous", topic, list(all_keywords)
            )
            added = 0
            for kw in new_kws:
                if kw not in all_keywords:
                    all_keywords.add(kw)
                    added += 1
            print(f"added {added} (total: {len(all_keywords)})")
            time.sleep(random.uniform(0.5, 1.5))

    # Sort and write
    sorted_keywords = sorted(all_keywords)
    with open(output_path, "w") as f:
        f.write(f"# Generated keywords — {len(sorted_keywords)} unique keywords across {len(TOPIC_TREE)} categories\n")
        f.write(f"# Generated by generate_keywords.py\n\n")
        for kw in sorted_keywords:
            f.write(f"{kw}\n")

    print(f"\n{'='*60}")
    print(f"DONE: {len(sorted_keywords)} unique keywords saved to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
