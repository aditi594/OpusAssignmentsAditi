"""
models/personalization.py
--------------------------
Rule-based offer recommendation engine.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Offer:
    title:    str
    description: str
    category: str
    priority: int

CATALOG = {
    "High": [
        Offer("Platinum Rewards Credit Card",
              "Unlimited 3% cashback on all purchases. No annual fee for first year.",
              "credit_card", 1),
        Offer("Exclusive Low-APR Personal Loan",
              "Loans up to $100K at rates as low as 5.9% APR.",
              "loan", 2),
        Offer("Priority Concierge Banking",
              "Dedicated relationship manager and priority queue access.",
              "premium", 3),
        Offer("Wealth Management Advisory",
              "Complimentary portfolio review and investment planning session.",
              "investment", 4),
    ],
    "Medium": [
        Offer("Standard Cashback Card",
              "1.5% cashback on everyday purchases with no minimum spend.",
              "credit_card", 1),
        Offer("Home Improvement Loan",
              "Fixed-rate loans up to $30K for renovation projects.",
              "loan", 2),
        Offer("Savings Account Upgrade",
              "Boost savings rate to 4.5% APY — limited time offer.",
              "savings", 3),
        Offer("Referral Bonus",
              "Earn $50 for each friend you refer who opens an account.",
              "engagement", 4),
    ],
    "Low": [
        Offer("Starter Cashback Card",
              "No annual fee, 1% cashback, credit-builder program.",
              "credit_card", 1),
        Offer("Financial Wellness Program",
              "Free budgeting tools, webinars, and credit score tracking.",
              "engagement", 2),
        Offer("Spend-to-Unlock Rewards",
              "Spend $200/month to unlock 2x cashback the following month.",
              "cashback", 3),
        Offer("Re-engagement Bonus",
              "$25 credit when you make 5 transactions in the next 30 days.",
              "engagement", 4),
    ],
}

from models.offer_manager import load_offers

def get_offers(segment, top_n=4):
    offers = load_offers()
    return [o for o in offers if o["segment"] == segment and o["active"]][:top_n]


def get_offer_rationale(segment: str, clv: float, frequency: int, avg_spend: float) -> str:
    if segment == "High":
        return (f"This customer has a predicted CLV of **${clv:,.0f}** and transacts "
                f"**{frequency}×/month** with an average spend of **${avg_spend:.0f}**. "
                "Premium products maximise engagement and long-term revenue retention.")
    elif segment == "Medium":
        return (f"CLV of **${clv:,.0f}** with moderate frequency ({frequency}×/month). "
                "Targeted offers can migrate this customer to the High-value tier.")
    else:
        return (f"CLV of **${clv:,.0f}**. Activation and engagement offers can build "
                "loyalty and increase transaction frequency over time.")
