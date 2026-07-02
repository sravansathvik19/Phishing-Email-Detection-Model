"""
generate_dataset.py
Generates a synthetic but realistic dataset of phishing and legitimate emails.
In a real-world project, you would replace this with a real dataset such as:
- Kaggle "Phishing Email Detection" datasets
- Nazario Phishing Corpus
- Enron dataset (for legitimate emails)
"""

import pandas as pd
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Phishing email templates (common social-engineering patterns)
# ---------------------------------------------------------------------------
phishing_templates = [
    "Dear customer, your account has been suspended. Click here {url} to verify your identity immediately.",
    "URGENT: Your bank account will be locked in 24 hours. Login now at {url} to avoid suspension.",
    "Congratulations! You have won a $1000 gift card. Claim your prize now at {url} before it expires.",
    "We detected unusual activity on your PayPal account. Verify your information at {url} to restore access.",
    "Your password will expire today. Update your credentials immediately at {url} to keep your account active.",
    "Security Alert: Someone tried to log into your account. Confirm your identity here {url} now.",
    "Dear user, your Apple ID has been locked due to suspicious activity. Unlock it at {url}.",
    "Your package could not be delivered. Pay a small fee at {url} to reschedule delivery.",
    "Action required: Your email storage is full. Click {url} to upgrade your account for free.",
    "We need you to confirm your billing details at {url} or your subscription will be cancelled.",
    "Your tax refund of $850 is ready. Submit your bank details at {url} to receive payment.",
    "Final Notice: Your account access will be terminated. Verify now at {url} to prevent suspension.",
    "Hi, this is IT support. We need your password to fix an urgent security issue. Reply or visit {url}.",
    "You have a new voicemail message. Click {url} to listen now using your email login.",
    "Your Netflix payment failed. Update your payment method at {url} to avoid losing access.",
    "Dear employee, please review and sign this urgent document at {url} before end of day.",
    "Your Microsoft 365 account requires re-verification. Click {url} immediately to avoid data loss.",
    "Claim your inheritance of $2,000,000 by providing your bank details at {url}.",
    "Your social security number has been suspended due to suspicious activity, call us or visit {url}.",
    "Limited time offer just for you! Get 90% off at {url}, verify your card now to claim discount.",
]

phishing_urls = [
    "http://secure-paypal-verification.com/login",
    "http://account-update-bank.net/verify",
    "http://bit.ly/3xJkLm9",
    "http://apple-id-locked-support.com",
    "http://192.168.45.22/login.php",
    "http://win-prize-claim-now.info",
    "http://tinyurl.com/freegift123",
    "http://netflix-billing-update.co",
    "http://microsoft365-reverify.support",
    "http://irs-tax-refund-claim.com",
]

# ---------------------------------------------------------------------------
# Legitimate email templates
# ---------------------------------------------------------------------------
legit_templates = [
    "Hi team, please find attached the minutes from yesterday's meeting. Let me know if I missed anything.",
    "Hey, are we still on for lunch tomorrow at 1pm? Let me know if that works for you.",
    "Your order #{order} has shipped and is expected to arrive in 3-5 business days. Track it at {url}.",
    "Thank you for your purchase. Your receipt is attached for your records.",
    "Reminder: The quarterly report is due this Friday. Please submit your section by end of day Thursday.",
    "Hi, just checking in to see how the project is going. Let me know if you need any help.",
    "Here are the slides from today's presentation as promised. Feel free to share with your team.",
    "Your subscription has been renewed successfully. No action is needed from your end.",
    "Looking forward to catching up this weekend! Let me know what time works for you.",
    "Please review the attached invoice for last month's services and let me know if you have questions.",
    "Our office will be closed on Monday for the public holiday. Normal operations resume Tuesday.",
    "Congratulations on completing the course! Your certificate is attached to this email.",
    "Hi, I've updated the shared document with the latest figures, take a look when you get a chance.",
    "Your flight booking is confirmed. Check-in opens 24 hours before departure.",
    "Just a friendly reminder that your appointment is scheduled for next Tuesday at 10am.",
    "Thanks for attending the webinar today. The recording is available at {url} on our official site.",
    "We appreciate your feedback and will use it to improve our service going forward.",
    "Here's the agenda for tomorrow's stand-up meeting, see you all at 9am.",
    "Your monthly statement is now available in your account dashboard.",
    "Welcome to the team! Here's some information to help you get started in your first week.",
]

legit_urls = [
    "https://www.amazon.com/orders",
    "https://www.dropbox.com/shared/abc123",
    "https://company.zoom.us/recording/share",
    "https://github.com/company/project",
    "https://www.linkedin.com/in/profile",
    "https://mail.google.com/inbox",
    "https://www.company-portal.com/dashboard",
]


def make_phishing_email():
    template = random.choice(phishing_templates)
    url = random.choice(phishing_urls)
    text = template.format(url=url)
    return text


def make_legit_email():
    template = random.choice(legit_templates)
    url = random.choice(legit_urls)
    order = random.randint(10000, 99999)
    text = template.format(url=url, order=order)
    return text


def generate_dataset(n_phishing=400, n_legit=400):
    rows = []
    for _ in range(n_phishing):
        rows.append({"text": make_phishing_email(), "label": "phishing"})
    for _ in range(n_legit):
        rows.append({"text": make_legit_email(), "label": "legitimate"})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("emails.csv", index=False)
    print(f"Generated dataset with {len(df)} emails")
    print(df["label"].value_counts())
