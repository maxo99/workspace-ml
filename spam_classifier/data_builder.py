import pandas as pd
from sklearn.model_selection import train_test_split


def create_sample_dataset():
    # Sample spam messages
    spam_messages = [
        "WINNER! You've won a $1000 prize! Click here now!",
        "Congratulations! You've been selected for a free iPhone!",
        "URGENT: Your account needs verification. Click immediately!",
        "Make money fast! Work from home! Guaranteed income!",
        "FREE VIAGRA! Limited time offer! Order now!",
        "You've inherited $10 million! Contact us immediately!",
        "Hot singles in your area! Click here to meet them!",
        "Lose weight fast with this one weird trick!",
        "CONGRATULATIONS! You're our lucky winner today!",
        "Get rich quick! This is not a scam! Guaranteed!",
        "FREE MONEY! No strings attached! Click now!",
        "Your PayPal account has been suspended! Verify now!",
        "AMAZING OFFER! Buy one get ten free! Limited stock!",
        "You've been pre-approved for a loan! Apply now!",
        "CLICK HERE FOR FREE PRIZES! Don't miss out!",
    ] * 10  # Repeat to get more samples

    # Sample legitimate messages (ham)
    ham_messages = [
        "Hi, can we schedule a meeting for tomorrow at 2pm?",
        "Thanks for your email. I'll review the document and get back to you.",
        "The project deadline is next Friday. Please submit your work by then.",
        "Could you please send me the quarterly report when you have a chance?",
        "Great job on the presentation! The client was very impressed.",
        "Reminder: Team lunch is scheduled for Thursday at noon.",
        "I've attached the files you requested. Let me know if you need anything else.",
        "The meeting has been rescheduled to next week. I'll send an updated invite.",
        "Please review the attached contract and let me know your thoughts.",
        "Thank you for your support on this project. Much appreciated!",
        "Can you help me with this issue when you get a chance?",
        "The system maintenance is scheduled for this weekend.",
        "I've completed the analysis. Here are my findings.",
        "Let's discuss the new features in our next standup.",
        "Your order has been shipped and should arrive by Friday.",
    ] * 10  # Repeat to get more samples

    # Combine messages and create labels
    all_messages = spam_messages + ham_messages
    labels = [1] * len(spam_messages) + [0] * len(ham_messages)

    # Create DataFrame
    df = pd.DataFrame({"text": all_messages, "label": labels})

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def split_data(df) -> tuple:
    return train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
