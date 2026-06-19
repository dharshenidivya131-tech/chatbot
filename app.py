# -*- coding: utf-8 -*-

import random
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data = [
("I am very happy today", "joy"),
("I feel sad and lonely", "sadness"),
("I am scared about exams", "fear"),
("I am angry and frustrated", "anger"),
("I love my life", "joy"),
("I feel nothing special", "surprise"),
("I am stressed", "fear"),
("I feel low", "sadness"),
("I am excited", "joy"),
("I feel okay", "surprise")
]

texts = [t[0] for t in data]
labels = [t[1] for t in data]

model = Pipeline([
("tfidf", TfidfVectorizer()),
("clf", LogisticRegression(max_iter=1000))
])

model.fit(texts, labels)

mood_map = {
"joy": "Happy",
"sadness": "Sad",
"fear": "Anxious",
"anger": "Angry",
"surprise": "Neutral"
}

responses = {
"Happy": [
"That's wonderful to hear! Keep embracing the positive moments.",
"Your happiness is inspiring. Keep up the great energy!",
"I'm glad you're feeling good today. Keep smiling!"
],


"Sad": [
    "I'm sorry you're feeling down. Remember that difficult moments are temporary.",
    "It's okay to feel sad sometimes. Be kind to yourself.",
    "Take things one step at a time. You are stronger than you think."
],

"Anxious": [
    "Take a deep breath. Focus on the present moment.",
    "Try breaking your tasks into smaller steps. You've got this.",
    "Remember that it's okay to ask for help when you need it."
],

"Angry": [
    "It's okay to feel angry sometimes. Take a moment to pause and breathe.",
    "Try stepping away from the situation and giving yourself some time to cool down.",
    "Focus on what you can control and let go of what you cannot change."
],

"Neutral": [
    "Thank you for sharing your thoughts.",
    "How has your day been so far?",
    "I'm here to listen whenever you'd like to talk."
]


}

tips = [
"Practice deep breathing for 5 minutes.",
"Take a short walk and stretch your body.",
"Listen to calming music.",
"Drink a glass of water and relax.",
"Write down three things you're grateful for.",
"Try a 5-minute meditation session.",
"Take a short break from screens.",
"Spend a few minutes outdoors in fresh air.",
"Talk to a trusted friend or family member.",
"Focus on one positive thing that happened today."
]

def chatbot(message):
    raw_mood = model.predict([message])[0]
    mood = mood_map.get(raw_mood, "Neutral")

    return f"""
Detected Mood: {mood}

Response:
{random.choice(responses[mood])}

Wellness Tip:
{random.choice(tips)}
"""
interface = gr.Interface(
fn=chatbot,
inputs=gr.Textbox(
lines=4,
placeholder="How are you feeling today?"
),
outputs=gr.Textbox(lines=10),
title="🧠 Mental Health Chatbot",
description="Share your feelings and receive supportive responses with wellness tips."
)


interface.launch(share=True)
