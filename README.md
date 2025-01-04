# Question Generation Project

This project generates different types of questions—Multiple Choice Questions (MCQ), Fill-in-the-Blanks, and Matching Questions—based on a given context.

## 🚀 **Setup and Installation**

### 1. **Clone the Repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. **Install Dependencies**
Make sure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

### 3. **Download Pre-trained Models**
This project relies on pre-trained models for generating different types of questions. Download them from the links below:

- **BERT-WSD (Matching Questions)**: [Download Here](https://drive.google.com/file/d/1--1hkVvJrRVzPiMnK_5ejD4LKivIpQyF/view?usp=drive_link)
- **Distractor Generation Model**: [Download Here](https://drive.google.com/file/d/1_7I3EpkaTDZDVxCO4I3oJBIjsokN0f_k/view?usp=drive_link)
- **MCQ and Answer Generation Model**: [Download Here](https://drive.google.com/file/d/1-Jl_11rwFf5f7GH9EPhZ5GZ68w45yHcf/view?usp=drive_link)
- **Sense2Vec Model (Kaggle)**: [Download Here](https://www.kaggle.com/pedromoya/sense2vec-reddit-2015-dataset-spacy-model)

## 📚 **Usage**

### 1. **Import the Question Generator Class**
```python
from app.question_generator import QuestionGenerator
```

### 2. **Initialize the Generator**
```python
context = "Your input text here"
generator = QuestionGenerator(context)
```

### 3. **Generate Questions**
You can specify the type of questions you want:
- `mcq`: Multiple Choice Questions
- `fill_in_blank`: Fill-in-the-Blanks
- `matching`: Matching Questions

```python
questions = generator.generate_questions(['mcq', 'fill_in_blank', 'matching'])
print(questions)
```
