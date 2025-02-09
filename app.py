from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://Sudha:VishalSona%4013122003@localhost/flashcards_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Flashcard Model
class Flashcard(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(255), nullable=False)
    answer = db.Column(db.String(255), nullable=False)
    review_count = db.Column(db.Integer, default=0)

# Create Tables
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return "Welcome to the AI Flashcard System!"

@app.route('/flashcards', methods=['POST'])
def create_flashcards():
    data = request.json
    if 'question' not in data or 'answer' not in data:
        return jsonify({'error': 'Question and answer are required'}), 400
    
    flashcard = Flashcard(question=data['question'], answer=data['answer'])
    db.session.add(flashcard)
    db.session.commit()
    return jsonify({'message': 'Flashcard added successfully'}), 201

@app.route('/review', methods=['GET'])
def get_flashcards_for_review():
    if request.args.get('load') != 'true':
        return jsonify([])
    flashcards = Flashcard.query.all()
    return jsonify([
        {'id': f.id, 'question': f.question, 'answer': f.answer, 'review_count': f.review_count} 
        for f in flashcards
    ])

@app.route('/review/<int:id>', methods=['POST'])
def update_flashcard_review(id):
    card = Flashcard.query.get(id)
    if not card:
        return jsonify({'message': 'Flashcard not found'}), 404
    
    card.review_count += 1
    db.session.commit()
    return jsonify({'message': 'Review updated', 'review_count': card.review_count})

@app.route('/analytics', methods=['GET'])
def review_analytics():
    flashcards = Flashcard.query.order_by(Flashcard.review_count.desc()).all()
    return jsonify([{'question': f.question, 'review_count': f.review_count} for f in flashcards])

@app.route('/ai-suggestions', methods=['GET'])
def ai_suggestions():
    flashcards = Flashcard.query.all()
    questions = [f.question for f in flashcards]

    if len(questions) < 2:
        return jsonify({'message': 'Not enough flashcards for AI suggestions'}), 400

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)

    suggestions = []
    for idx, card in enumerate(flashcards):
        similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-3:][::-1]
        suggestions.append({
            'question': card.question,
            'suggested': [questions[i] for i in top_indices if i != idx]
        })

    return jsonify(suggestions)

@app.route('/clear-flashcards', methods=['DELETE'])
def clear_flashcards():
    try:
        num_deleted = db.session.query(Flashcard).delete()
        db.session.commit()
        return jsonify({'message': f'{num_deleted} flashcards deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

