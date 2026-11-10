"""
Generates a labelled SMS/email-style dataset (data/synthetic_spam.csv) with columns: label, text.

This is a SYNTHETIC dataset so the project runs anywhere without downloads.
For your final submission, replace data/spam.csv with the real "SMS Spam Collection"
dataset (UCI / Kaggle). Rename its columns to `label` (ham/spam) and `text` and
the rest of the pipeline works unchanged.
"""
import os
import random

import pandas as pd

random.seed(42)

# ----------------------------- SPAM building blocks -----------------------------
spam_openers = [
    "CONGRATULATIONS!", "WINNER!!", "Urgent!", "Dear customer,", "Hello friend,",
    "FREE entry:", "Final notice:", "Act now!", "You have been selected!", "Alert:",
    "Limited time offer!", "Hi,", "Attention:", "Claim now!",
]
spam_bodies = [
    "you have won a ${amt} cash prize. Call {phone} to claim your reward now",
    "your account has been suspended. Verify your details at {url} immediately",
    "get a FREE {gadget} today. Click {url} to claim, offer ends tonight",
    "you are our lucky winner of a brand new {gadget}. Reply WIN to {short} now",
    "earn ${amt} per week working from home. No experience needed. Visit {url}",
    "your loan of ${amt} is pre-approved at 0% interest. Apply now at {url}",
    "buy cheap meds online without prescription. Huge discount, order at {url}",
    "your bank password expires today. Update it at {url} to avoid account closure",
    "claim your FREE ringtone and 100 free texts. Text STOP to {short} to opt out",
    "you have been chosen for an exclusive investment deal. Double your money in 7 days. Call {phone}",
    "a package is waiting for you. Pay the small delivery fee of ${small} at {url}",
    "hot singles in your area want to meet you. Click {url} for free access",
    "lowest price guaranteed on {gadget}. 90% OFF, buy now at {url} before stock runs out",
    "your lottery ticket won ${amt}! Send your bank details to claim the prize",
    "cheap insurance quotes, save up to 70%. Get your free quote at {url}",
    "unlock unlimited credit card offers. No credit check. Apply at {url} today",
]
spam_closers = [
    "", "T&Cs apply.", "Reply STOP to unsubscribe.", "Hurry, offer expires soon!",
    "18+ only.", "Don't miss out!!!", "Call now!", "100% guaranteed.", "No obligation.",
]

# ----------------------------- HAM building blocks -----------------------------
ham_templates = [
    "hey are you coming to {place} today?",
    "ok see you at {time} then",
    "can you send me the notes from {subject} class?",
    "i will be late, stuck in traffic near {place}",
    "happy birthday {name}! have a great day",
    "mom said dinner is ready, come home soon",
    "don't forget the {subject} assignment is due tomorrow",
    "lol that was so funny, tell me more later",
    "thanks for helping me yesterday, really appreciate it",
    "what time does the {subject} lecture start tomorrow?",
    "i'm at the library, want to join me for the project work?",
    "let's meet at {place} at {time}, I'll bring the books",
    "call me when you reach home",
    "did you watch the match last night? it was amazing",
    "can you pick up some milk and bread on your way back?",
    "sorry I missed your call, I was in the lab. will call you back",
    "the meeting is moved to {time}. please confirm if you can attend",
    "good morning! have a nice day at college",
    "I've uploaded the report to the shared folder, please review it",
    "are we still on for lunch at {place}?",
    "need help with the {subject} code, it keeps throwing an error",
    "reminder: team project submission is on friday",
    "ok no problem, take your time",
    "I'll text you the address, it's near {place}",
    "wow congratulations {name}, I heard you got placed! party soon",
    "can we reschedule our call to {time}? something came up",
    "please send me your presentation slides before the review",
    "just reached the hostel, will message you in the evening",
    "how was your exam today? hope it went well",
    "bring your id card tomorrow, they will check at the gate",
]

fill = {
    "amt": ["1000", "5000", "10000", "50000", "1,000,000", "250"],
    "small": ["1.99", "2.50", "5"],
    "phone": ["0800 123 4567", "09061701461", "+1-800-555-0199", "87077", "0207 946 0000"],
    "short": ["80062", "85023", "63333", "87121"],
    "url": ["http://bit.ly/win-now", "www.claim-prize.biz", "http://secure-login.info/verify",
            "www.free-offers.xyz", "http://cheap-deals.top/buy", "www.cash-today.club"],
    "gadget": ["iPhone", "laptop", "smart TV", "PlayStation", "Samsung phone", "gift card"],
    "place": ["the canteen", "the library", "the main gate", "the cafe", "college", "the station"],
    "time": ["5pm", "6:30", "10am", "noon", "tomorrow morning", "8 tonight"],
    "subject": ["maths", "physics", "python", "DBMS", "networks", "chemistry"],
    "name": ["Riya", "Aman", "Priya", "Rahul", "Sneha", "Karan"],
}


def fill_text(t: str) -> str:
    for k, vals in fill.items():
        while "{" + k + "}" in t:
            t = t.replace("{" + k + "}", random.choice(vals), 1)
    return t


def noisy(text: str) -> str:
    """Small random noise so the data isn't perfectly clean."""
    r = random.random()
    if r < 0.15:
        text = text.lower()
    elif r < 0.25:
        text = text.upper()
    return text


def make_spam() -> str:
    parts = [random.choice(spam_openers), fill_text(random.choice(spam_bodies)),
             random.choice(spam_closers)]
    return noisy(" ".join(p for p in parts if p))


def make_ham() -> str:
    t = fill_text(random.choice(ham_templates))
    if random.random() < 0.3:  # some hams also contain urls / numbers / money words
        t += " " + random.choice(["", "ok?", "thanks", "see the link http://college.edu/notes",
                                  "it costs 50 rupees only", "call me at 9876543210"])
    return noisy(t)


def main(n_ham=1700, n_spam=600, out="data/synthetic_spam.csv"):
    rows = [("ham", make_ham()) for _ in range(n_ham)] + \
           [("spam", make_spam()) for _ in range(n_spam)]
    random.shuffle(rows)
    df = pd.DataFrame(rows, columns=["label", "text"]).drop_duplicates()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
