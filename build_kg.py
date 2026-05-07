"""Minimal KG builder template for Assignment 4.

Keep this contract unchanged:
- Graph: (Regulation)-[:HAS_ARTICLE]->(Article)-[:CONTAINS_RULE]->(Rule)
- Article: number, content, reg_name, category
- Rule: rule_id, type, action, result, art_ref, reg_name
- Fulltext indexes: article_content_idx, rule_idx
- SQLite file: ncu_regulations.db
"""

import os
import json
import re
import sqlite3
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)


def extract_entities(article_number: str, reg_name: str, content: str) -> dict[str, Any]:
    """Extract structured rules from one article using the local model."""
    load_local_llm()
    tok = get_tokenizer()
    pipe = get_raw_pipeline()
    if tok is None or pipe is None:
        return {"rules": []}

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert system for extracting atomic rules from university regulations.\n\n"
                "OUTPUT FORMAT (STRICT JSON ONLY):\n"
                "{\"rules\": [{\"type\": string, \"action\": string, \"result\": string}]}\n\n"
                "CORE PRINCIPLES:\n"
                "1. Each rule MUST contain exactly ONE action and ONE result.\n"
                "2. NEVER combine multiple actions into one rule.\n"
                "3. NEVER combine multiple penalties/results into one rule.\n"
                "4. If a sentence contains multiple actions (connected by 'and', 'or', commas), split them.\n"
                "5. If a sentence contains multiple penalties (e.g., zero grade AND disciplinary action), split them into separate rules.\n"
                "6. Each penalty (zero grade, point deduction, disciplinary action, report, etc.) MUST produce its own rule.\n"
                "7. Preserve all explicit numeric values exactly (e.g., 30 minutes, 100 NTD).\n\n"
                "IMPORTANT SPLITTING RULES:\n"
                "- 'A or B or C' → 3 separate rules\n"
                "- 'A and B' → 2 separate rules\n"
                "- 'Violation results in X and Y' → 2 separate rules\n"
                "- 'Violation results in zero grade and disciplinary action' → MUST output TWO rules\n\n"
                "GOOD EXAMPLE:\n"
                "Text: Students must not talk or copy answers. Violators shall receive zero grade and be reported.\n"
                "Output:\n"
                "{\n"
                "  \"rules\": [\n"
                "    {\"type\": \"prohibition\", \"action\": \"talk during exam\", \"result\": \"not allowed\"},\n"
                "    {\"type\": \"prohibition\", \"action\": \"copy others' answers\", \"result\": \"not allowed\"},\n"
                "    {\"type\": \"penalty\", \"action\": \"talk during exam\", \"result\": \"zero grade\"},\n"
                "    {\"type\": \"penalty\", \"action\": \"talk during exam\", \"result\": \"reported to authority\"},\n"
                "    {\"type\": \"penalty\", \"action\": \"copy others' answers\", \"result\": \"zero grade\"},\n"
                "    {\"type\": \"penalty\", \"action\": \"copy others' answers\", \"result\": \"reported to authority\"}\n"
                "  ]\n"
                "}\n\n"
                "BAD EXAMPLE (DO NOT DO THIS):\n"
                "{\n"
                "  \"rules\": [\n"
                "    {\"type\": \"penalty\", \"action\": \"talk or copy answers\", \"result\": \"zero grade and reported\"}\n"
                "  ]\n"
                "}\n\n"
                "FEE EXAMPLE (IMPORTANT):\n"
                "If different fees exist for different cases, output separate rules.\n"
                "Example:\n"
                "\"EasyCard replacement fee is 200 NTD, Mifare card is 100 NTD\"\n"
                "→ MUST output TWO rules.\n\n"
                "Be concise, factual, and strictly follow the schema. No explanations, no markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Regulation: {reg_name}\n"
                f"Article: {article_number}\n"
                "Extraction focus:\n"
                "- Capture each distinct requirement/penalty/fee/time-limit as one rule.\n"
                "- When fees differ by card type, produce one rule per card type.\n"
                "- Keep numbers exact (e.g., 200 NTD, 100 NTD, 3 working days).\n\n"
                f"Text:\n{content}"
            ),
        },
    ]
    try:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        raw = pipe(prompt, max_new_tokens=340)[0]["generated_text"].strip()
    except Exception:
        return {"rules": []}

    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        pass

    if parsed is None:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None

    if not isinstance(parsed, dict):
        return {"rules": []}

    rules = parsed.get("rules", [])
    if not isinstance(rules, list):
        return {"rules": []}

    cleaned: list[dict[str, str]] = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        r_type = str(r.get("type", "general")).strip() or "general"
        action = str(r.get("action", "")).strip()
        result = str(r.get("result", "")).strip()
        if not action or not result:
            continue
        cleaned.append({"type": r_type, "action": action, "result": result})

    # Guard against LLM over-merging fee facts in one sentence.
    expanded: list[dict[str, str]] = []
    for rule in cleaned:
        r_type = str(rule.get("type", "")).lower().strip()
        action = str(rule.get("action", "")).strip()
        result = str(rule.get("result", "")).strip()

        text_block = f"{action} {result}".lower()
        amounts = re.findall(r"(\d+)\s*(?:ntd|dollars?)", text_block)
        has_easy = ("easycard" in text_block) or ("悠遊卡" in text_block)
        has_mifare = "mifare" in text_block

        if r_type == "fee" and len(amounts) >= 2 and has_easy and has_mifare:
            easy_amount = amounts[0]
            mifare_amount = amounts[1]
            expanded.append(
                {
                    "type": "fee",
                    "action": "Student ID replacement (EasyCard)",
                    "result": f"Fee is {easy_amount} NTD",
                }
            )
            expanded.append(
                {
                    "type": "fee",
                    "action": "Student ID replacement (Mifare)",
                    "result": f"Fee is {mifare_amount} NTD",
                }
            )
            continue

        expanded.append(rule)

    return {"rules": expanded}


def build_fallback_rules(article_number: str, content: str) -> list[dict[str, str]]:
    """Deterministic fallback extractor for critical numeric/penalty facts."""
    text = content.strip()
    low = text.lower()
    rules: list[dict[str, str]] = []

    minutes = re.findall(r"(\d+)\s*(?:minutes?|mins?)", low)
    points = re.findall(r"(\d+)\s*points?", low)
    currency = re.findall(r"(\d+)\s*(?:ntd|dollars?)", low)
    credits = re.findall(r"(\d+)\s*credits?", low)
    years = re.findall(r"(\d+)\s*years?", low)
    semesters = re.findall(r"(\d+)\s*semesters?", low)

    if "late" in low and minutes:
        rules.append(
            {
                "type": "time_limit",
                "action": "Student arrival after exam starts",
                "result": f"Late beyond {minutes[0]} minutes is not allowed",
            }
        )

    if ("leave" in low or "exit" in low) and minutes:
        rules.append(
            {
                "type": "time_limit",
                "action": "Leaving exam room after exam starts",
                "result": f"Cannot leave until {minutes[-1]} minutes have passed",
            }
        )

    if ("penalty" in low or "deduction" in low or "deduct" in low) and points:
        rules.append(
            {
                "type": "penalty",
                "action": "Violation of exam rules",
                "result": f"{points[0]} points deduction",
            }
        )

    if "zero" in low or "zero score" in low:
        rules.append(
            {
                "type": "penalty",
                "action": "Serious exam misconduct",
                "result": "Zero score",
            }
        )

    if "disciplinary" in low:
        rules.append(
            {
                "type": "discipline",
                "action": "Serious exam misconduct",
                "result": "Subject to disciplinary action",
            }
        )

    if "fee" in low and currency:
        rules.append(
            {
                "type": "fee",
                "action": "Student ID replacement",
                "result": f"Fee is {currency[0]} NTD",
            }
        )

    if "working days" in low and re.search(r"(\d+)\s*working\s*days", low):
        d = re.search(r"(\d+)\s*working\s*days", low)
        if d:
            rules.append(
                {
                    "type": "time_limit",
                    "action": "Processing student ID replacement",
                    "result": f"Issued in {d.group(1)} working days",
                }
            )

    if ("graduation" in low or "graduate" in low) and credits:
        rules.append(
            {
                "type": "graduation_requirement",
                "action": "Undergraduate graduation",
                "result": f"Requires at least {credits[0]} credits",
            }
        )

    if "physical education" in low and semesters:
        rules.append(
            {
                "type": "graduation_requirement",
                "action": "Undergraduate PE requirement",
                "result": f"Requires {semesters[0]} semesters of PE",
            }
        )

    if ("duration" in low or "study" in low) and years:
        rules.append(
            {
                "type": "duration",
                "action": "Bachelor study duration",
                "result": f"Standard duration is {years[0]} years",
            }
        )

    if "extension" in low and years:
        rules.append(
            {
                "type": "duration",
                "action": "Maximum undergraduate extension",
                "result": f"Extension up to {years[-1]} years",
            }
        )

    if "passing score" in low and re.search(r"(\d+)\s*points?", low):
        p = re.findall(r"(\d+)\s*points?", low)
        if p:
            rules.append(
                {
                    "type": "grading",
                    "action": "Course passing threshold",
                    "result": f"Passing score is {p[0]} points",
                }
            )

    if "leave of absence" in low and years:
        rules.append(
            {
                "type": "leave",
                "action": "Maximum leave of absence",
                "result": f"Up to {years[-1]} academic years",
            }
        )

    if "make-up exam" in low and ("no" in low or "not" in low):
        rules.append(
            {
                "type": "exam_policy",
                "action": "Request make-up exam for failed course",
                "result": "Not allowed",
            }
        )

    return rules


# SQLite tables used:
# - regulations(reg_id, name, category)
# - articles(reg_id, article_number, content)


def build_graph() -> None:
    """Build KG from SQLite into Neo4j using the fixed assignment schema."""
    sql_conn = sqlite3.connect("ncu_regulations.db")
    cursor = sql_conn.cursor()
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Optional: warm up local LLM
    load_local_llm()

    with driver.session() as session:
        # Fixed strategy: clear existing graph data before rebuilding.
        session.run("MATCH (n) DETACH DELETE n")

        # 1) Read regulations and create Regulation nodes.
        cursor.execute("SELECT reg_id, name, category FROM regulations")
        regulations = cursor.fetchall()
        reg_map: dict[int, tuple[str, str]] = {}

        for reg_id, name, category in regulations:
            reg_map[reg_id] = (name, category)
            session.run(
                "MERGE (r:Regulation {id:$rid}) SET r.name=$name, r.category=$cat",
                rid=reg_id,
                name=name,
                cat=category,
            )

        # 2) Read articles and create Article + HAS_ARTICLE.
        cursor.execute("SELECT reg_id, article_number, content FROM articles")
        articles = cursor.fetchall()

        for reg_id, article_number, content in articles:
            reg_name, reg_category = reg_map.get(
                reg_id, ("Unknown", "Unknown"))
            session.run(
                """
                MATCH (r:Regulation {id: $rid})
                CREATE (a:Article {
                    number:   $num,
                    content:  $content,
                    reg_name: $reg_name,
                    category: $reg_category
                })
                MERGE (r)-[:HAS_ARTICLE]->(a)
                """,
                rid=reg_id,
                num=article_number,
                content=content,
                reg_name=reg_name,
                reg_category=reg_category,
            )

        # 3) Create full-text index on Article content.
        session.run(
            """
            CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS
            FOR (a:Article) ON EACH [a.content]
            """
        )

        rule_counter = 0

        seen_rule_keys: set[str] = set()

        for reg_id, article_number, content in articles:
            reg_name, _ = reg_map.get(reg_id, ("Unknown", "Unknown"))
            extracted = extract_entities(article_number, reg_name, content)
            llm_rules = extracted.get("rules", []) if isinstance(
                extracted, dict) else []
            fallback_rules = build_fallback_rules(article_number, content)

            merged_rules: list[dict[str, str]] = []
            if isinstance(llm_rules, list):
                merged_rules.extend(llm_rules)
            merged_rules.extend(fallback_rules)

            for rule in merged_rules:
                if not isinstance(rule, dict):
                    continue

                r_type = str(rule.get("type", "general")).strip() or "general"
                action = str(rule.get("action", "")).strip()
                result = str(rule.get("result", "")).strip()

                if not action or not result:
                    continue

                key = "|".join(
                    [
                        reg_name.lower().strip(),
                        article_number.lower().strip(),
                        r_type.lower().strip(),
                        re.sub(r"\s+", " ", action.lower()),
                        re.sub(r"\s+", " ", result.lower()),
                    ]
                )
                if key in seen_rule_keys:
                    continue
                seen_rule_keys.add(key)

                rule_counter += 1
                rule_id = f"R{rule_counter:06d}"

                session.run(
                    """
                    MATCH (a:Article {number:$num, reg_name:$reg_name, content:$content})
                    MERGE (r:Rule {rule_id:$rule_id})
                    SET r.type=$type,
                        r.action=$action,
                        r.result=$result,
                        r.art_ref=$art_ref,
                        r.reg_name=$reg_name
                    MERGE (a)-[:CONTAINS_RULE]->(r)
                    """,
                    num=article_number,
                    reg_name=reg_name,
                    content=content,
                    rule_id=rule_id,
                    type=r_type,
                    action=action,
                    result=result,
                    art_ref=article_number,
                )

        # 4) Create full-text index on Rule fields.
        session.run(
            """
            CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS
            FOR (r:Rule) ON EACH [r.action, r.result]
            """
        )

        # 5) Coverage audit (provided scaffold).
        coverage = session.run(
            """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:CONTAINS_RULE]->(r:Rule)
            WITH a, count(r) AS rule_count
            RETURN count(a) AS total_articles,
                   sum(CASE WHEN rule_count > 0 THEN 1 ELSE 0 END) AS covered_articles,
                   sum(CASE WHEN rule_count = 0 THEN 1 ELSE 0 END) AS uncovered_articles
            """
        ).single()

        total_articles = int((coverage or {}).get("total_articles", 0) or 0)
        covered_articles = int(
            (coverage or {}).get("covered_articles", 0) or 0)
        uncovered_articles = int(
            (coverage or {}).get("uncovered_articles", 0) or 0)

        print(
            f"[Coverage] covered={covered_articles}/{total_articles}, "
            f"uncovered={uncovered_articles}"
        )

    driver.close()
    sql_conn.close()


if __name__ == "__main__":
    build_graph()
