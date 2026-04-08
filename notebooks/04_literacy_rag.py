# Databricks notebook source
# ArthaSetu v2 — Financial Literacy RAG Pipeline
# FAISS vector search + Param-1 (or LLM fallback) for financial education
# Writes quiz engagement data to Gold tables

# COMMAND ----------

import json
import os
import numpy as np
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import *

CATALOG = "arthasetu"
GOLD_SCHEMA = f"{CATALOG}.xscore_gold"

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# LITERACY CORPUS — 6 Core Modules
# ═══════════════════════════════════════════════════════════════════════

LITERACY_CORPUS = {
    "LIT_001": {
        "title": "Understanding Credit Scores",
        "topic": "credit",
        "difficulty": 3,
        "xscore_component": "financial_awareness",
        "chunks": [
            "A credit score is a three-digit number that represents your creditworthiness. In India, CIBIL score ranges from 300 to 900. Banks use this score to decide whether to give you a loan and at what interest rate. A score above 750 is considered good.",
            "Your credit score is affected by several factors: payment history (35%), credit utilization (30%), length of credit history (15%), types of credit (10%), and new credit inquiries (10%). Paying your EMIs on time is the single most important factor.",
            "Even if you don't have a traditional credit history, your financial behavior matters. Regular bill payments, consistent savings, and responsible UPI usage all indicate creditworthiness. This is what alternative credit scoring like XScore captures.",
            "To build a good credit score: (1) Always pay EMIs and credit card bills on time, (2) Keep credit utilization below 30%, (3) Don't apply for too many loans at once, (4) Maintain a mix of secured and unsecured credit, (5) Check your credit report annually for errors.",
        ],
        "quiz": [
            {"question": "What is a good CIBIL score range?", "options": ["100-300", "300-500", "500-700", "750-900"], "answer": 3},
            {"question": "What factor most affects your credit score?", "options": ["Income", "Payment history", "Age", "Education"], "answer": 1},
            {"question": "What should your credit utilization be below?", "options": ["10%", "30%", "50%", "80%"], "answer": 1},
        ]
    },
    "LIT_002": {
        "title": "EMI & Loan Basics",
        "topic": "loans",
        "difficulty": 4,
        "xscore_component": "financial_awareness",
        "chunks": [
            "EMI stands for Equated Monthly Installment. It is the fixed amount you pay every month to repay a loan. EMI consists of two parts: principal (the actual loan amount) and interest (the bank's charge for lending). In the early months of a loan, more of your EMI goes toward interest. As time passes, more goes toward principal.",
            "The EMI formula is: EMI = P × r × (1+r)^n / ((1+r)^n - 1), where P is the principal amount, r is the monthly interest rate (annual rate ÷ 12 ÷ 100), and n is the loan tenure in months. For example, a ₹1 lakh loan at 12% annual rate for 1 year gives an EMI of ₹8,885.",
            "Before taking a loan, calculate your debt-to-income ratio. Your total EMIs should not exceed 40% of your monthly income. If you earn ₹20,000/month, your total EMIs should stay below ₹8,000. Going above this makes you a high-risk borrower.",
            "Types of loans in India: (1) Personal loans — unsecured, higher interest (12-24%), (2) Home loans — secured by property, lower interest (8-10%), (3) Gold loans — secured by gold, quick disbursement, (4) Microfinance loans — small amounts for SHG members, (5) Mudra loans — government scheme for small business, up to ₹10 lakh.",
        ],
        "quiz": [
            {"question": "What does EMI stand for?", "options": ["Equal Monthly Income", "Equated Monthly Installment", "Easy Money Investment", "Electronic Monthly Interest"], "answer": 1},
            {"question": "Your total EMIs should not exceed what % of income?", "options": ["20%", "40%", "60%", "80%"], "answer": 1},
            {"question": "Which loan type typically has the lowest interest rate?", "options": ["Personal loan", "Home loan", "Credit card", "Payday loan"], "answer": 1},
        ]
    },
    "LIT_003": {
        "title": "Savings & Budgeting",
        "topic": "savings",
        "difficulty": 2,
        "xscore_component": "financial_stability",
        "chunks": [
            "The 50-30-20 rule is a simple budgeting framework: 50% of income for needs (rent, food, utilities), 30% for wants (entertainment, dining out, shopping), and 20% for savings and debt repayment. Even on a ₹10,000 salary, saving ₹2,000/month builds a strong financial foundation.",
            "An emergency fund should cover 3-6 months of expenses. This protects you from unexpected costs like medical bills or job loss. Start small — even ₹500/month adds up. Keep your emergency fund in a savings account or liquid mutual fund for easy access.",
            "Government savings schemes for Indian citizens: (1) PPF — 15-year lock-in, tax-free returns, (2) Sukanya Samriddhi — for girl child education, (3) PMSBY — ₹12/year accident insurance, (4) PMJJBY — ₹436/year life insurance, (5) Atal Pension Yojana — pension of ₹1,000-5,000/month after 60.",
            "Small habits that improve savings: (1) Track all UPI transactions weekly, (2) Set up auto-debit for savings on salary day, (3) Use the ₹100 rule — before any purchase over ₹100, wait 24 hours, (4) Cook at home more often — save ₹3,000-5,000/month, (5) Review subscriptions monthly.",
        ],
        "quiz": [
            {"question": "In the 50-30-20 rule, how much should go to savings?", "options": ["10%", "20%", "30%", "50%"], "answer": 1},
            {"question": "How many months of expenses should an emergency fund cover?", "options": ["1-2", "3-6", "8-10", "12+"], "answer": 1},
            {"question": "Which scheme is for girl child education?", "options": ["PPF", "Sukanya Samriddhi", "PMSBY", "NPS"], "answer": 1},
        ]
    },
    "LIT_004": {
        "title": "UPI Safety & Digital Payments",
        "topic": "upi_safety",
        "difficulty": 2,
        "xscore_component": "digital_trust",
        "chunks": [
            "UPI (Unified Payments Interface) processes over 10 billion transactions per month in India. While UPI is safe by design (encrypted, RBI-regulated), users must follow safety practices. Never share your UPI PIN with anyone — just like you would never share your ATM PIN.",
            "Common UPI fraud methods: (1) Fake payment requests — scammer sends a 'collect' request pretending to be a refund, (2) Screen sharing scams — asking you to install AnyDesk/TeamViewer, (3) Fake customer care numbers, (4) QR code manipulation — modified QR that debits instead of credits. Rule: You NEVER need to enter PIN to receive money.",
            "Safe UPI practices: (1) Only use official bank apps or verified UPI apps, (2) Enable app lock and device lock, (3) Check the receiver's name before paying, (4) Set daily transaction limits, (5) Report fraud immediately via NPCI helpline (14431) or your bank, (6) Never scan QR codes from unknown sources.",
            "Benefits of digital payments for your financial profile: Regular UPI usage creates a transaction history that can help build your credit profile. Consistent merchant payments show financial stability. Bill payments via UPI are automatically tracked, building your payment discipline record.",
        ],
        "quiz": [
            {"question": "Do you need to enter PIN to receive money via UPI?", "options": ["Yes", "No", "Sometimes", "Only for large amounts"], "answer": 1},
            {"question": "What is the NPCI helpline number for UPI fraud?", "options": ["100", "14431", "1800", "112"], "answer": 1},
            {"question": "Which is NOT a UPI safety practice?", "options": ["Enable app lock", "Share PIN with family", "Check receiver name", "Set transaction limits"], "answer": 1},
        ]
    },
    "LIT_005": {
        "title": "Bill Payment Importance",
        "topic": "budgeting",
        "difficulty": 3,
        "xscore_component": "payment_discipline",
        "chunks": [
            "On-time bill payment is the strongest signal of creditworthiness. When you pay your electricity, water, and phone bills on time consistently, it shows financial discipline. This behavior directly improves your XScore payment discipline component, which carries the highest weight.",
            "Late payments have hidden costs beyond the late fee: (1) Service disconnection risk, (2) Reconnection charges, (3) Impact on credit profile, (4) Stress and time spent resolving issues, (5) Loss of auto-pay discounts. A ₹50 late fee on a ₹500 bill is effectively a 10% penalty.",
            "Tips for never missing a bill: (1) Set up calendar reminders 3 days before due date, (2) Use auto-debit via UPI/NACH for recurring bills, (3) Pay bills immediately upon receiving them, (4) Keep a separate 'bills fund' equal to one month of total bills, (5) Use bill payment apps that send reminders.",
            "Payment streaks matter: Paying all bills on time for 6 consecutive months is a powerful credit signal. Banks and lenders look at consistency, not just individual payments. Even one missed payment breaks your streak and can take months to rebuild trust.",
        ],
        "quiz": [
            {"question": "Which bill payment behavior most improves creditworthiness?", "options": ["Paying large amounts", "Consistent on-time payment", "Paying in cash", "Paying yearly"], "answer": 1},
            {"question": "What is the recommended approach for recurring bills?", "options": ["Pay when convenient", "Set up auto-debit", "Wait for final notice", "Pay quarterly"], "answer": 1},
            {"question": "How long does a good payment streak need to be significant?", "options": ["1 month", "3 months", "6 months", "1 year"], "answer": 2},
        ]
    },
    "LIT_006": {
        "title": "Understanding Interest Rates",
        "topic": "interest_rates",
        "difficulty": 4,
        "xscore_component": "financial_awareness",
        "chunks": [
            "Interest is the cost of borrowing money. When a bank charges 12% annual interest on a ₹1 lakh loan, you pay ₹12,000 per year in interest alone. The total repayment = principal + interest. Understanding this prevents over-borrowing and helps you compare loan offers.",
            "Flat rate vs reducing balance: A 10% flat rate on ₹1 lakh for 1 year means you pay ₹10,000 interest. But a 10% reducing balance rate means interest decreases as you pay EMIs — total interest is only about ₹5,500. Always ask: 'Is this flat rate or reducing balance?' Reducing balance is always cheaper.",
            "The power of compound interest works for savings too. ₹1,000/month invested at 8% annual return grows to: ₹1.5 lakh in 10 years, ₹5.8 lakh in 20 years, ₹14.9 lakh in 30 years. Starting early is more important than investing large amounts.",
            "Red flags in lending: (1) Interest rate above 36% annual — may be predatory, (2) Processing fees above 3%, (3) No written agreement, (4) Pressure to borrow more than needed, (5) Hidden charges not disclosed upfront. Always calculate the total cost of a loan before signing.",
        ],
        "quiz": [
            {"question": "Which interest rate type is cheaper for borrowers?", "options": ["Flat rate", "Reducing balance", "Both are same", "Variable rate"], "answer": 1},
            {"question": "What annual interest rate may indicate a predatory loan?", "options": ["10%", "18%", "24%", "Above 36%"], "answer": 3},
            {"question": "₹1,000/month at 8% for 30 years grows to approximately?", "options": ["₹3.6 lakh", "₹7.2 lakh", "₹14.9 lakh", "₹36 lakh"], "answer": 2},
        ]
    },
}

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# BUILD FAISS VECTOR INDEX
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Building FAISS Vector Index for RAG")
print("═" * 60)

try:
    from sentence_transformers import SentenceTransformer
    import faiss

    # Load embedding model
    print("  Loading embedding model (MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Prepare chunks
    all_chunks = []
    chunk_metadata = []

    for mod_id, mod in LITERACY_CORPUS.items():
        for i, chunk_text in enumerate(mod["chunks"]):
            all_chunks.append(chunk_text)
            chunk_metadata.append({
                "module_id": mod_id,
                "title": mod["title"],
                "topic": mod["topic"],
                "difficulty": mod["difficulty"],
                "chunk_index": i,
            })

    print(f"  Embedding {len(all_chunks)} chunks...")
    embeddings = embed_model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)

    # Build FAISS index
    dimension = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexIVFFlat(
        faiss.IndexFlatIP(dimension),  # Inner product (cosine similarity on normalized vectors)
        dimension,
        min(4, len(all_chunks)),  # nlist
    )
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 2  # Search 2 clusters

    print(f"  ✓ FAISS index built: {index.ntotal} vectors, {dimension} dimensions")

except ImportError as e:
    print(f"  [WARN] {e}")
    print("  Install: pip install sentence-transformers faiss-cpu")
    embed_model = None
    index = None

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# RAG RETRIEVAL + LLM GENERATION
# ═══════════════════════════════════════════════════════════════════════

def retrieve_context(query, top_k=3):
    """Retrieve top-k relevant chunks for a query."""
    if embed_model is None or index is None:
        return [], []

    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)

    contexts = []
    metadata = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(all_chunks):
            contexts.append(all_chunks[idx])
            metadata.append({**chunk_metadata[idx], "similarity_score": float(score)})

    return contexts, metadata


def generate_response(query, contexts, user_language="en", user_literacy_level="medium"):
    """Generate a response using retrieved context + LLM."""

    context_text = "\n\n".join(contexts)

    # Adjust complexity based on literacy level
    complexity_guide = {
        "low": "Use very simple Hindi/English words. Short sentences. Give practical examples from daily life.",
        "medium": "Use clear language with some financial terms. Explain concepts with relatable examples.",
        "high": "Use proper financial terminology. Include numbers and calculations where relevant.",
    }

    prompt = f"""You are a financial literacy educator for rural and semi-urban India.
Answer the user's question using ONLY the context below. Be concise, practical, and empathetic.
Language level: {complexity_guide.get(user_literacy_level, complexity_guide['medium'])}

Context:
{context_text}

User question: {query}

Answer in 2-3 short paragraphs. Include one actionable tip."""

    # Try Databricks-hosted LLM first, then fallback
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        # Try Databricks Foundation Model API
        response = w.serving_endpoints.query(
            name="databricks-meta-llama-3-3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"  [LLM Fallback] Databricks LLM unavailable: {e}")
        # Return context-based summary as fallback
        return f"Based on our financial literacy content:\n\n{contexts[0]}\n\n💡 Tip: {contexts[-1].split('.')[-2].strip()}."


def answer_financial_query(query, user_language="en", user_literacy_level="medium"):
    """End-to-end RAG pipeline for financial literacy queries."""
    print(f"  Query: {query}")

    # Retrieve
    contexts, metadata = retrieve_context(query, top_k=3)
    if not contexts:
        return "I'm sorry, I don't have information about that topic yet."

    print(f"  Retrieved {len(contexts)} chunks from modules: {[m['module_id'] for m in metadata]}")

    # Generate
    response = generate_response(query, contexts, user_language, user_literacy_level)

    return response


# Test RAG pipeline
print("\n── Testing RAG Pipeline ──")
test_queries = [
    "What is a credit score and why does it matter?",
    "How do I calculate my EMI?",
    "How to save money on a small salary?",
    "Is UPI safe to use?",
]

for query in test_queries:
    print(f"\n{'─' * 40}")
    response = answer_financial_query(query)
    print(f"  Response: {response[:200]}...")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# QUIZ ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_quiz(module_id, user_id):
    """Run a quiz for a given module and record engagement."""
    if module_id not in LITERACY_CORPUS:
        return {"error": f"Module {module_id} not found"}

    module = LITERACY_CORPUS[module_id]
    quiz = module["quiz"]
    results = []

    print(f"\n  📝 Quiz: {module['title']} ({len(quiz)} questions)")

    # Simulate quiz responses (in real app, this would be interactive)
    for i, q in enumerate(quiz):
        # Random simulation — in real app, user answers
        user_answer = q["answer"]  # Simulating correct answer
        is_correct = user_answer == q["answer"]
        results.append(is_correct)

    score = int(sum(results) / len(results) * 100)

    engagement = {
        "user_id": user_id,
        "module_id": module_id,
        "topic_category": module["topic"],
        "quiz_score": score,
        "completion_flag": True,
        "time_spent_seconds": 600,  # Simulated
        "attempts": 1,
        "difficulty_level": module["difficulty"],
        "session_timestamp": datetime.now().isoformat(),
        "language_used": "en",
    }

    return engagement


# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# WRITE CORPUS TO DELTA FOR DLT PIPELINE
# ═══════════════════════════════════════════════════════════════════════

print("\n── Saving Literacy Corpus to Delta ──")

corpus_rows = []
for mod_id, mod in LITERACY_CORPUS.items():
    for i, chunk in enumerate(mod["chunks"]):
        corpus_rows.append({
            "chunk_id": f"{mod_id}_chunk_{i:02d}",
            "doc_id": mod_id,
            "title": mod["title"],
            "topic": mod["topic"],
            "difficulty": mod["difficulty"],
            "text": chunk,
            "chunk_index": i,
        })

corpus_df = spark.createDataFrame(corpus_rows)

(corpus_df
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.xscore_bronze.literacy_raw_content")
)

print(f"  ✓ {len(corpus_rows)} literacy chunks written to bronze.literacy_raw_content")

# Also save quiz data
quiz_rows = []
for mod_id, mod in LITERACY_CORPUS.items():
    for i, q in enumerate(mod["quiz"]):
        quiz_rows.append({
            "quiz_id": f"{mod_id}_quiz_{i:02d}",
            "module_id": mod_id,
            "question": q["question"],
            "options": json.dumps(q["options"]),
            "correct_answer_index": q["answer"],
        })

quiz_df = spark.createDataFrame(quiz_rows)
(quiz_df
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.xscore_bronze.literacy_quiz_bank")
)

print(f"  ✓ {len(quiz_rows)} quiz questions written to bronze.literacy_quiz_bank")

print("\n✓ Literacy RAG Pipeline Complete!")
