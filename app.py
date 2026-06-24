from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# --- MASSIVE PROFESSIONAL KNOWLEDGE BASE ---
faqs = [
    # AI GENERAL
    {"question": "What is Artificial Intelligence?", "answer": "AI is the simulation of human intelligence by machines, especially computer systems. It involves learning, reasoning, and self-correction."},
    {"question": "What is the difference between AI and ML?", "answer": "AI is the broad concept of machines being able to carry out tasks in a smart way. Machine Learning is an application of AI based around the idea that we should give machines access to data and let them learn for themselves."},
    {"question": "What is Strong vs Weak AI?", "answer": "Weak AI (Narrow AI) is designed for a specific task (like Siri). Strong AI (AGI) is a theoretical AI that has the mental capabilities of a human."},
    
    # MACHINE LEARNING
    {"question": "What is Machine Learning?", "answer": "Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed."},
    {"question": "What is Supervised Learning?", "answer": "Supervised learning is where the model is trained on a labeled dataset (it knows the answers during training). Examples include Linear Regression and Classification."},
    {"question": "What is Unsupervised Learning?", "answer": "Unsupervised learning involves training on data that has no labels. The system tries to learn the patterns and structure by itself (Clustering)."},
    {"question": "What is Regression?", "answer": "Regression is a type of Supervised Learning used to predict continuous numerical values, like house prices or stock trends."},
    {"question": "What is Classification?", "answer": "Classification is a type of Supervised Learning used to categorize data into specific classes, like 'Spam' or 'Not Spam'."},
    {"question": "What is Overfitting?", "answer": "Overfitting happens when a model learns the training data 'too well,' including the noise, and fails to perform accurately on new, unseen data."},

    # DATA SCIENCE
    {"question": "What is Data Science?", "answer": "Data Science is a field that combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data."},
    {"question": "What is Big Data?", "answer": "Big Data refers to extremely large datasets that may be analyzed computationally to reveal patterns, trends, and associations."},
    {"question": "What is a Data Scientist?", "answer": "A Data Scientist is a professional who uses data to help organizations make better decisions through analysis, visualization, and predictive modeling."},

    # DEEP LEARNING & NLP
    {"question": "What is Deep Learning?", "answer": "Deep Learning is a subset of ML based on Artificial Neural Networks with multiple layers. it is used for high-level tasks like face recognition."},
    {"question": "What are Neural Networks?", "answer": "Neural Networks are a series of algorithms that endeavor to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates."},
    {"question": "What is NLP?", "answer": "Natural Language Processing (NLP) allows computers to understand and respond to text or voice data in much the same way human beings do."},
    {"question": "What is a Chatbot?", "answer": "A chatbot is an AI-powered software designed to simulate a conversation with human users, especially over the internet."},
    
    # INTERNSHIP SPECIFIC
    {"question": "What are the CodeAlpha tasks?", "answer": "The tasks include Language Translation, FAQ Chatbot, Music Generation, and Object Detection. You are currently looking at Task 2!"}
]
# --- THE LOGIC ---
questions = [f["question"].lower() for f in faqs]
answers = [f["answer"] for f in faqs]

vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(questions)

def get_best_answer(user_query):
    # Clean user input
    user_query = user_query.lower().strip()
    
    # Convert query to vector
    user_vec = vectorizer.transform([user_query])
    
    # Find similarity
    similarities = cosine_similarity(user_vec, question_vectors).flatten()
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]

    # PROFESSIONAL LOGIC
    # If the score is too low, the user asked something outside our AI/ML topic.
    if best_score < 0.3:
        return "I am a specialized AI & Data Science assistant. I don't have information on that specific query. Please ask me about ML, Deep Learning, or Data Science basics!"
        
    return answers[best_index]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_msg = data.get("question", "")
    answer = get_best_answer(user_msg)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)