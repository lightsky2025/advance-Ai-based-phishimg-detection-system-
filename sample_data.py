"""
Sample Data Generator for Phishing Detection System
Generates sample emails and URLs for testing and demonstration
"""

import json
import random
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

# Phishing email templates
PHISHING_EMAILS = [
    {
        "subject": "URGENT: Your account has been compromised!",
        "body": "Dear valued customer,\n\nWe have detected suspicious activity on your account. Your account will be suspended within 24 hours unless you verify your identity immediately.\n\nClick here to verify: http://secure-verify-account.xyz/login\n\nFailure to act will result in permanent account termination.\n\nSecurity Team",
        "sender": "security-alert@gmail.com",
        "label": 1,
        "description": "Account suspension threat with urgent verification link"
    },
    {
        "subject": "Congratulations! You've won $1,000,000!",
        "body": "Dear Winner,\n\nYou have been selected as our lucky winner of $1,000,000 in our annual lottery!\n\nTo claim your prize, please provide:\n- Full Name\n- Address\n- Social Security Number\n- Bank Account Details\n\nClaim here: http://prize-claim-center.tk/claim\n\nCongratulations again!\nPrize Committee",
        "sender": "lottery-winner@prize-center.ga",
        "label": 1,
        "description": "Lottery scam requesting personal information"
    },
    {
        "subject": "Action Required: Update your payment information",
        "body": "Hello,\n\nYour payment method on file has expired. Please update your payment information to avoid service interruption.\n\nUpdate now: http://billing-update-secure.ml/payment\n\nIf you don't update within 48 hours, your account will be charged a late fee.\n\nBilling Department",
        "sender": "billing@payment-update.xyz",
        "label": 1,
        "description": "Payment update phishing with urgency"
    },
    {
        "subject": "Your PayPal account has been limited",
        "body": "Dear PayPal User,\n\nWe've temporarily limited your PayPal account due to suspicious activity.\n\nPlease confirm your identity to restore full access: http://paypal-verify-account.ga/verify\n\nIf you don't respond within 24 hours, your account will be permanently suspended.\n\nPayPal Security",
        "sender": "security@paypa1-verify.com",
        "label": 1,
        "description": "PayPal impersonation scam"
    },
    {
        "subject": "FINAL NOTICE: Overdue Invoice #45892",
        "body": "URGENT NOTICE\n\nYou have an overdue invoice of $1,234.56 that requires immediate payment.\n\nView and pay your invoice: http://invoice-payment-portal.xyz/invoice/45892\n\nFailure to pay will result in legal action and credit score damage.\n\nAccounts Receivable",
        "sender": "collections@invoice-dept.tk",
        "label": 1,
        "description": "Fake invoice scam with legal threats"
    }
]

# Legitimate email templates
LEGITIMATE_EMAILS = [
    {
        "subject": "Weekly Team Meeting - Monday 10 AM",
        "body": "Hi Team,\n\nJust a reminder about our weekly standup meeting tomorrow at 10 AM.\n\nAgenda:\n- Sprint progress updates\n- Blockers and challenges\n- Upcoming deadlines\n\nPlease come prepared with your updates.\n\nBest regards,\nJohn Smith\nTeam Lead",
        "sender": "john.smith@company.com",
        "label": 0,
        "description": "Standard team meeting reminder"
    },
    {
        "subject": "Q4 Project Report - Review Requested",
        "body": "Hello,\n\nI've completed the Q4 project report and would appreciate your feedback.\n\nThe report covers:\n- Project milestones achieved\n- Budget analysis\n- Resource allocation\n- Next quarter planning\n\nPlease review and provide your comments by Friday.\n\nThanks,\nSarah Johnson\nProject Manager",
        "sender": "sarah.johnson@company.com",
        "label": 0,
        "description": "Professional project report request"
    },
    {
        "subject": "Lunch tomorrow?",
        "body": "Hey!\n\nA few of us are planning to grab lunch tomorrow at the new Italian place on Main Street.\n\nWant to join? We're thinking around 12:30 PM.\n\nLet me know!\nMike",
        "sender": "mike.wilson@company.com",
        "label": 0,
        "description": "Casual lunch invitation"
    },
    {
        "subject": "IT Security Update - Password Policy Changes",
        "body": "Dear All,\n\nAs part of our ongoing security improvements, we're updating our password policy effective next month.\n\nKey changes:\n- Minimum password length: 12 characters\n- Require special characters\n- Password expiration: 90 days\n\nDetailed guidelines will be shared via the company intranet.\n\nIT Security Team",
        "sender": "it-security@company.com",
        "label": 0,
        "description": "Official IT security announcement"
    },
    {
        "subject": "Welcome to the Company!",
        "body": "Welcome to the team!\n\nWe're excited to have you join us. Your first day orientation will be on Monday at 9 AM in the main conference room.\n\nPlease bring:\n- Government-issued ID\n- Completed I-9 form\n- Direct deposit information\n\nLooking forward to working with you!\n\nHR Department",
        "sender": "hr@company.com",
        "label": 0,
        "description": "New employee welcome email"
    }
]

# Phishing URLs
PHISHING_URLS = [
    {"url": "http://paypa1-verify.ga/login", "description": "Fake PayPal login"},
    {"url": "http://secure-bank-verify.xyz/account", "description": "Fake bank verification"},
    {"url": "http://amazon-prize-claim.tk/winner", "description": "Fake Amazon prize"},
    {"url": "http://microsoft-support-help.ml/fix", "description": "Fake Microsoft support"},
    {"url": "http://apple-id-locked.cf/unlock", "description": "Fake Apple ID unlock"},
    {"url": "http://netflix-billing-update.xyz/payment", "description": "Fake Netflix billing"},
    {"url": "http://irs-tax-refund.ga/claim", "description": "Fake IRS refund"},
    {"url": "http://fedex-package-tracking.tk/track", "description": "Fake FedEx tracking"},
    {"url": "http://google-security-alert.ml/verify", "description": "Fake Google security"},
    {"url": "http://facebook-account-recover.xyz/restore", "description": "Fake Facebook recovery"}
]

# Legitimate URLs
LEGITIMATE_URLS = [
    {"url": "https://www.google.com", "description": "Google homepage"},
    {"url": "https://github.com/explore", "description": "GitHub explore page"},
    {"url": "https://stackoverflow.com/questions", "description": "Stack Overflow questions"},
    {"url": "https://www.linkedin.com/feed", "description": "LinkedIn feed"},
    {"url": "https://docs.python.org/3/", "description": "Python documentation"},
    {"url": "https://aws.amazon.com/console/", "description": "AWS console"},
    {"url": "https://en.wikipedia.org/wiki/Main_Page", "description": "Wikipedia main page"},
    {"url": "https://www.youtube.com", "description": "YouTube homepage"},
    {"url": "https://medium.com", "description": "Medium homepage"},
    {"url": "https://www.reddit.com", "description": "Reddit homepage"}
]


def generate_sample_emails(n_phishing: int = 10, n_legitimate: int = 10) -> List[Dict]:
    """Generate sample email data for testing"""
    samples = []
    
    # Generate phishing samples
    for i in range(n_phishing):
        template = random.choice(PHISHING_EMAILS).copy()
        # Add some variation
        template["id"] = f"phish_{i+1}"
        template["generated_at"] = datetime.now().isoformat()
        samples.append(template)
    
    # Generate legitimate samples
    for i in range(n_legitimate):
        template = random.choice(LEGITIMATE_EMAILS).copy()
        template["id"] = f"legit_{i+1}"
        template["generated_at"] = datetime.now().isoformat()
        samples.append(template)
    
    random.shuffle(samples)
    return samples


def generate_sample_urls(n_phishing: int = 5, n_legitimate: int = 5) -> List[Dict]:
    """Generate sample URL data for testing"""
    samples = []
    
    # Generate phishing URLs
    for i in range(n_phishing):
        template = random.choice(PHISHING_URLS).copy()
        template["id"] = f"phish_url_{i+1}"
        template["generated_at"] = datetime.now().isoformat()
        samples.append(template)
    
    # Generate legitimate URLs
    for i in range(n_legitimate):
        template = random.choice(LEGITIMATE_URLS).copy()
        template["id"] = f"legit_url_{i+1}"
        template["generated_at"] = datetime.now().isoformat()
        samples.append(template)
    
    random.shuffle(samples)
    return samples


def save_sample_data(output_dir: str = "data"):
    """Save sample data to JSON files"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save emails
    emails = generate_sample_emails(20, 20)
    with open(f"{output_dir}/sample_emails.json", "w") as f:
        json.dump(emails, f, indent=2)
    
    # Generate and save URLs
    urls = generate_sample_urls(10, 10)
    with open(f"{output_dir}/sample_urls.json", "w") as f:
        json.dump(urls, f, indent=2)
    
    # Generate CSV format for training
    email_csv_lines = ["subject,body,sender,label,description"]
    for email in emails:
        subject = email["subject"].replace(",", ";").replace("\n", " ")
        body = email["body"].replace(",", ";").replace("\n", " ")
        sender = email["sender"]
        label = email["label"]
        desc = email.get("description", "").replace(",", ";")
        email_csv_lines.append(f'"{subject}","{body}","{sender}",{label},"{desc}"')
    
    with open(f"{output_dir}/sample_emails.csv", "w") as f:
        f.write("\n".join(email_csv_lines))
    
    print(f"Sample data saved to {output_dir}/")
    print(f"  - sample_emails.json ({len(emails)} emails)")
    print(f"  - sample_urls.json ({len(urls)} URLs)")
    print(f"  - sample_emails.csv ({len(emails)} emails)")


if __name__ == "__main__":
    save_sample_data()