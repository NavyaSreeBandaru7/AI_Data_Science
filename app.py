# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import random
import re
from PIL import Image
import io
import base64
import textwrap
import time
import os

# Page configuration
st.set_page_config(
    page_title="AI & Data Science Learning Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'user_progress' not in st.session_state:
    st.session_state.user_progress = {
        'completed_lessons': [],
        'quiz_scores': {},
        'projects_completed': [],
        'skill_level': 'Beginner',
        'job_applications': [],
        'mind_maps': {}
    }

if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff6e40;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .project-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s;
    }
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .mind-map-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Learning content database
LEARNING_MODULES = {
    "Beginner": {
        "Python Fundamentals": {
            "topics": {
                "Variables & Data Types": "Understanding basic data types and variables in Python",
                "Control Flow": "Conditional statements and loops",
                "Functions": "Creating and using functions",
                "Data Structures": "Lists, tuples, dictionaries, sets"
            },
            "duration": "2 weeks",
            "projects": ["Calculator App", "To-Do List Manager"],
            "resources": ["Python Documentation", "Interactive Tutorial"]
        },
        "Data Science Basics": {
            "topics": {
                "NumPy": "Numerical computing with arrays",
                "Pandas": "Data manipulation and analysis",
                "Data Visualization": "Creating plots with Matplotlib and Seaborn",
                "Statistics": "Descriptive and inferential statistics"
            },
            "duration": "3 weeks",
            "projects": ["EDA on Titanic Dataset", "Sales Data Analysis"],
            "resources": ["Pandas Cheat Sheet", "Visualization Gallery"]
        },
        "Machine Learning Introduction": {
            "topics": {
                "Supervised Learning": "Regression and classification",
                "Model Evaluation": "Metrics and validation techniques",
                "Feature Engineering": "Preprocessing and transformation",
                "Scikit-learn": "Implementing ML algorithms"
            },
            "duration": "4 weeks",
            "projects": ["House Price Prediction", "Iris Classification"],
            "resources": ["Scikit-learn Documentation", "ML Tutorials"]
        }
    },
    "Intermediate": {
        "Advanced ML": {
            "topics": {
                "Ensemble Methods": "Random Forests, Gradient Boosting",
                "Hyperparameter Tuning": "Grid search and random search",
                "Dimensionality Reduction": "PCA, t-SNE",
                "Model Deployment": "Basic deployment techniques"
            },
            "duration": "4 weeks",
            "projects": ["Customer Churn Prediction", "Credit Risk Assessment"],
            "resources": ["Advanced ML Techniques", "Deployment Guide"]
        },
        "Deep Learning": {
            "topics": {
                "Neural Networks": "Perceptrons and activation functions",
                "CNNs": "Image classification",
                "RNNs": "Sequence modeling",
                "Transfer Learning": "Using pre-trained models"
            },
            "duration": "6 weeks",
            "projects": ["Image Classification", "Text Sentiment Analysis"],
            "resources": ["TensorFlow Tutorials", "PyTorch Examples"]
        },
        "NLP Fundamentals": {
            "topics": {
                "Text Processing": "Tokenization, stemming, lemmatization",
                "Word Embeddings": "Word2Vec, GloVe",
                "Named Entity Recognition": "Identifying entities in text",
                "Topic Modeling": "LDA and NMF"
            },
            "duration": "4 weeks",
            "projects": ["Spam Detection", "Document Clustering"],
            "resources": ["NLP with Python", "Transformers Guide"]
        }
    },
    "Advanced": {
        "Advanced Deep Learning": {
            "topics": {
                "GANs": "Generative Adversarial Networks",
                "Autoencoders": "Dimensionality reduction and generation",
                "Transformers": "Attention mechanisms",
                "BERT/GPT": "State-of-the-art language models"
            },
            "duration": "8 weeks",
            "projects": ["Image Generation", "Custom Chatbot"],
            "resources": ["Research Papers", "Advanced Implementations"]
        },
        "MLOps": {
            "topics": {
                "Model Deployment": "Docker and Kubernetes",
                "CI/CD": "Continuous integration and deployment",
                "Monitoring": "Model performance tracking",
                "Scaling": "Distributed training"
            },
            "duration": "4 weeks",
            "projects": ["End-to-End ML Pipeline", "Model API Development"],
            "resources": ["MLOps Best Practices", "Cloud Platforms Guide"]
        },
        "Research & Innovation": {
            "topics": {
                "Research Papers": "Reading and implementing papers",
                "State-of-the-art Models": "Cutting-edge architectures",
                "Custom Architectures": "Designing novel models",
                "Experimentation": "Designing and running experiments"
            },
            "duration": "Ongoing",
            "projects": ["Research Paper Implementation", "Novel Model Development"],
            "resources": ["Academic Journals", "Conference Proceedings"]
        }
    }
}

# Quiz questions database
QUIZ_DATABASE = {
    "Python Fundamentals": [
        {
            "question": "What is the output of: print(type([1, 2, 3]))?",
            "options": ["<class 'list'>", "<class 'tuple'>", "<class 'dict'>", "<class 'set'>"],
            "correct": 0,
            "explanation": "In Python, square brackets [] denote a list."
        },
        {
            "question": "Which method is used to add an element to a list?",
            "options": ["add()", "append()", "insert_end()", "push()"],
            "correct": 1,
            "explanation": "The append() method adds an element to the end of a list."
        }
    ],
    "Machine Learning": [
        {
            "question": "Which metric is best for imbalanced classification?",
            "options": ["Accuracy", "F1-Score", "MSE", "MAE"],
            "correct": 1,
            "explanation": "F1-Score considers both precision and recall, making it suitable for imbalanced datasets."
        },
        {
            "question": "What does overfitting mean?",
            "options": [
                "Model performs poorly on training data",
                "Model performs well on training but poorly on test data",
                "Model performs well on both training and test data",
                "Model has too few parameters"
            ],
            "correct": 1,
            "explanation": "Overfitting occurs when a model learns the training data too well, including noise, and fails to generalize to new data."
        }
    ]
}

# Job application templates
JOB_TEMPLATES = {
    "Data Scientist": {
        "skills": ["Python", "Machine Learning", "Statistics", "SQL", "Data Visualization"],
        "keywords": ["predictive modeling", "statistical analysis", "A/B testing", "data pipeline"],
        "description": "Analyze complex data to help companies make decisions"
    },
    "ML Engineer": {
        "skills": ["Python", "TensorFlow/PyTorch", "MLOps", "Docker", "Cloud Platforms"],
        "keywords": ["model deployment", "scalability", "optimization", "production systems"],
        "description": "Build and deploy ML models at scale"
    },
    "Data Analyst": {
        "skills": ["SQL", "Excel", "Tableau/PowerBI", "Python/R", "Statistics"],
        "keywords": ["data insights", "reporting", "dashboards", "business intelligence"],
        "description": "Transform data into actionable insights"
    }
}

# Mind map database
MIND_MAP_DB = {
    "Machine Learning": {
        "Supervised Learning": "Learning from labeled data",
        "Unsupervised Learning": "Finding patterns in unlabeled data",
        "Reinforcement Learning": "Learning through rewards and penalties",
        "Deep Learning": "Neural networks with multiple layers"
    },
    "Data Science Process": {
        "Data Collection": "Gathering raw data",
        "Data Cleaning": "Handling missing values and outliers",
        "Exploratory Analysis": "Understanding data patterns",
        "Model Building": "Creating predictive models",
        "Deployment": "Putting models into production"
    }
}

def create_mind_map(topic, concepts):
    """Create an interactive mind map visualization"""
    fig = go.Figure()
    
    # Center node
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=50, color='#ff6e40'),
        text=[topic],
        textposition="middle center",
        textfont=dict(size=18, color='white', family="Arial Black"),
        hoverinfo='text',
        hovertext=topic
    ))
    
    # Concept nodes
    n = len(concepts)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    for i, (concept, details) in enumerate(concepts.items()):
        x = 2 * np.cos(angles[i])
        y = 2 * np.sin(angles[i])
        
        # Add edge
        fig.add_trace(go.Scatter(
            x=[0, x], y=[0, y],
            mode='lines',
            line=dict(color='#1e3d59', width=2),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add concept node
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=35, color='#1e3d59'),
            text=[concept],
            textposition="top center",
            textfont=dict(size=12, color='white'),
            hoverinfo='text',
            hovertext=f"<b>{concept}</b><br>{details}",
            showlegend=False
        ))
    
    fig.update_layout(
        showlegend=False,
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
    return fig

def analyze_resume_ats(resume_text, job_role):
    """Analyze resume for ATS compatibility"""
    template = JOB_TEMPLATES.get(job_role, JOB_TEMPLATES["Data Scientist"])
    
    # Check for keywords
    found_skills = []
    missing_skills = []
    
    for skill in template["skills"]:
        if re.search(rf'\b{re.escape(skill)}\b', resume_text, re.IGNORECASE):
            found_skills.append(skill)
        else:
            missing_skills.append(skill)
    
    # Check for action keywords
    found_keywords = []
    for keyword in template["keywords"]:
        if re.search(rf'\b{re.escape(keyword)}\b', resume_text, re.IGNORECASE):
            found_keywords.append(keyword)
    
    # Calculate ATS score
    skill_score = len(found_skills) / len(template["skills"]) * 50
    keyword_score = min(len(found_keywords) / len(template["keywords"]) * 50, 50)
    total_score = skill_score + keyword_score
    
    return {
        "score": total_score,
        "found_skills": found_skills,
        "missing_skills": missing_skills,
        "found_keywords": found_keywords,
        "recommendations": generate_recommendations(missing_skills, found_keywords, template["keywords"])
    }

def generate_recommendations(missing_skills, found_keywords, all_keywords):
    """Generate resume improvement recommendations"""
    recommendations = []
    
    if missing_skills:
        recommendations.append(f"Add these skills to your resume: {', '.join(missing_skills[:3])}")
    
    missing_keywords = [k for k in all_keywords if k not in found_keywords]
    if missing_keywords:
        recommendations.append(f"Include keywords like: {', '.join(missing_keywords[:3])}")
    
    if len(found_keywords) < 2:
        recommendations.append("Use more action verbs and industry-specific terminology")
    
    recommendations.append("Quantify your achievements with numbers and percentages")
    recommendations.append("Keep resume format simple and ATS-friendly (avoid complex formatting)")
    recommendations.append("Include relevant certifications and projects")
    recommendations.append("Tailor your resume for each job application")
    
    return recommendations

def generate_quiz(topic, num_questions=5):
    """Generate quiz questions for a topic"""
    # For demo, using predefined questions or generating random ones
    if topic in QUIZ_DATABASE:
        return QUIZ_DATABASE[topic][:num_questions]
    else:
        # Generate generic questions
        questions = []
        for i in range(num_questions):
            questions.append({
                "question": f"Sample question {i+1} about {topic}?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct": random.randint(0, 3),
                "explanation": f"This is an explanation for question {i+1}"
            })
        return questions

def calculate_learning_path(current_level, target_role):
    """Calculate personalized learning path"""
    path = []
    
    if current_level == "Beginner":
        path.extend(["Python Fundamentals", "Data Science Basics", "Machine Learning Introduction"])
    elif current_level == "Intermediate":
        path.extend(["Advanced ML", "Deep Learning"])
    
    # Add role-specific modules
    if "Engineer" in target_role:
        path.append("MLOps")
    elif "Scientist" in target_role:
        path.append("Advanced Statistics")
    elif "Analyst" in target_role:
        path.append("Business Intelligence")
    
    return path

def generate_cover_letter(company_name, position, user_skills):
    """Generate personalized cover letter"""
    return f"""
Dear Hiring Manager at {company_name},

I am writing to express my interest in the {position} position at {company_name}. 
With my background in {', '.join(user_skills[:3])} and passion for data-driven solutions, 
I am confident in my ability to contribute effectively to your team.

In my previous experience, I have successfully:
- Developed machine learning models that improved accuracy by 25%
- Implemented data pipelines processing 1M+ records daily
- Created interactive dashboards that informed key business decisions

I am particularly drawn to {company_name} because of your innovative approach to 
AI solutions and your commitment to [specific company value or project].

My attached resume provides further detail about my qualifications. 
I would welcome the opportunity to discuss how my skills and experiences 
align with the needs of your team.

Thank you for your time and consideration.

Sincerely,
[Your Name]
[Your Contact Information]
"""

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🎓 AI Learning Platform")
    
    menu = st.selectbox(
        "Navigation",
        ["Dashboard", "Learn", "Practice", "Projects", "Quizzes", 
         "Career Guide", "Resume Builder", "Mind Maps", "Progress"]
    )
    
    st.markdown("---")
    
    # User profile
    st.markdown("### 👤 User Profile")
    skill_level = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])
    st.session_state.user_progress['skill_level'] = skill_level
    
    target_role = st.selectbox(
        "Target Role",
        ["Data Scientist", "ML Engineer", "Data Analyst", "AI Researcher"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Completed Lessons", len(st.session_state.user_progress['completed_lessons']))
    st.metric("Projects Done", len(st.session_state.user_progress['projects_completed']))
    
    avg_score = np.mean(list(st.session_state.user_progress['quiz_scores'].values())) if st.session_state.user_progress['quiz_scores'] else 0
    st.metric("Avg Quiz Score", f"{avg_score:.1f}%")
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    if st.button("New Learning Session"):
        st.session_state.user_progress['completed_lessons'].append("New Session")
        st.success("Started new learning session!")
    if st.button("Generate Practice Exercise"):
        st.info("Generated new practice exercise!")

# Main content area
if menu == "Dashboard":
    st.markdown("<h1 class='main-header'>🚀 AI & Data Science Learning Platform</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Learning Modules")
        modules_count = sum(len(modules) for modules in LEARNING_MODULES.values())
        st.metric("Total Modules", modules_count)
        st.markdown("Comprehensive curriculum from basics to advanced")
    
    with col2:
        st.markdown("### 🎯 Projects")
        projects_count = sum(
            len(module_info.get("projects", [])) 
            for level_modules in LEARNING_MODULES.values() 
            for module_info in level_modules.values()
        )
        st.metric("Hands-on Projects", projects_count)
        st.markdown("Real-world projects to build your portfolio")
    
    with col3:
        st.markdown("### 💼 Career Support")
        st.metric("Job Roles Covered", len(JOB_TEMPLATES))
        st.markdown("Resume optimization and interview prep")
    
    # Learning path recommendation
    st.markdown("---")
    st.markdown("### 🗺️ Your Personalized Learning Path")
    
    learning_path = calculate_learning_path(skill_level, target_role)
    
    progress_cols = st.columns(len(learning_path))
    for i, module in enumerate(learning_path):
        with progress_cols[i]:
            if module in st.session_state.user_progress['completed_lessons']:
                st.success(f"✅ {module}")
            else:
                st.info(f"📘 {module}")
    
    # Recent achievements
    st.markdown("---")
    st.markdown("### 🏆 Recent Achievements")
    
    if st.session_state.user_progress['completed_lessons']:
        for lesson in st.session_state.user_progress['completed_lessons'][-3:]:
            st.markdown(f"- Completed: **{lesson}**")
    else:
        st.markdown("Start learning to earn achievements!")
    
    # Recommended next steps
    st.markdown("---")
    st.markdown("### 👣 Your Next Steps")
    st.markdown("1. Start with Python Fundamentals in the Learn section")
    st.markdown("2. Practice coding challenges in the Practice section")
    st.markdown("3. Build your first project: Titanic Survival Prediction")

elif menu == "Learn":
    st.markdown("<h1 class='main-header'>📚 Learning Modules</h1>", unsafe_allow_html=True)
    
    selected_level = st.selectbox("Select Level", ["Beginner", "Intermediate", "Advanced"])
    
    modules = LEARNING_MODULES[selected_level]
    
    for module_name, module_info in modules.items():
        with st.expander(f"📘 {module_name} - {module_info['duration']}"):
            st.markdown("**Topics Covered:**")
            for topic, description in module_info['topics'].items():
                st.markdown(f"#### {topic}")
                st.markdown(f"{description}")
            
            st.markdown("**Projects:**")
            for project in module_info['projects']:
                st.markdown(f"- 🛠️ {project}")
            
            st.markdown("**Resources:**")
            for resource in module_info['resources']:
                st.markdown(f"- 📚 {resource}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"Start Learning", key=f"learn_{module_name}"):
                    st.session_state.user_progress['completed_lessons'].append(module_name)
                    st.success(f"Started learning {module_name}!")
            
            with col2:
                if st.button(f"View Mind Map", key=f"mindmap_{module_name}"):
                    # Fixed: Store under user_progress
                    st.session_state.user_progress['mind_maps'][module_name] = module_info['topics']
            
            with col3:
                if st.button(f"Take Quiz", key=f"quiz_{module_name}"):
                    st.session_state.current_quiz = generate_quiz(module_name, 5)
                    st.experimental_rerun()

elif menu == "Practice":
    st.markdown("<h1 class='main-header'>💻 Practice Coding</h1>", unsafe_allow_html=True)
    
    practice_type = st.selectbox(
        "Select Practice Type",
        ["Python Basics", "Data Manipulation", "Machine Learning", "Deep Learning", "SQL"]
    )
    
    st.markdown("### 📝 Coding Challenge")
    
    challenges = {
        "Python Basics": {
            "title": "List Comprehension",
            "problem": "Create a list of squares for numbers 1 to 10 using list comprehension",
            "hint": "Use [x**2 for x in range(1, 11)]",
            "solution": "squares = [x**2 for x in range(1, 11)]"
        },
        "Data Manipulation": {
            "title": "Pandas DataFrame Operations",
            "problem": "Filter a DataFrame to show only rows where 'age' > 25 and 'salary' > 50000",
            "hint": "Use df[(df['age'] > 25) & (df['salary'] > 50000)]",
            "solution": "filtered_df = df[(df['age'] > 25) & (df['salary'] > 50000)]"
        },
        "Machine Learning": {
            "title": "Train-Test Split",
            "problem": "Split your data into 80% training and 20% testing sets",
            "hint": "Use train_test_split from sklearn.model_selection",
            "solution": "from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
        }
    }
    
    if practice_type in challenges:
        challenge = challenges[practice_type]
        st.markdown(f"**Challenge:** {challenge['title']}")
        st.markdown(f"**Problem:** {challenge['problem']}")
        
        code_input = st.text_area("Write your code here:", height=200)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Run Code"):
                st.success("Code executed successfully! (Simulation)")
                st.code("Output: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]")
        
        with col2:
            if st.button("Show Hint"):
                st.info(f"Hint: {challenge['hint']}")
        
        with col3:
            if st.button("View Solution"):
                st.code(challenge['solution'])

elif menu == "Projects":
    st.markdown("<h1 class='main-header'>🛠️ Hands-on Projects</h1>", unsafe_allow_html=True)
    
    project_category = st.selectbox(
        "Select Project Category",
        ["Beginner Projects", "Intermediate Projects", "Advanced Projects", "Portfolio Projects"]
    )
    
    projects = {
        "Beginner Projects": [
            {
                "name": "Titanic Survival Prediction",
                "description": "Predict passenger survival using logistic regression",
                "skills": ["Pandas", "Scikit-learn", "Data Visualization"],
                "difficulty": "⭐⭐",
                "duration": "2 days",
                "dataset": "Titanic passenger data"
            },
            {
                "name": "Stock Price Analysis",
                "description": "Analyze and visualize stock market trends",
                "skills": ["Pandas", "Matplotlib", "Time Series"],
                "difficulty": "⭐⭐",
                "duration": "3 days",
                "dataset": "Historical stock prices"
            }
        ],
        "Intermediate Projects": [
            {
                "name": "Customer Segmentation",
                "description": "Segment customers using clustering algorithms",
                "skills": ["K-Means", "PCA", "Feature Engineering"],
                "difficulty": "⭐⭐⭐",
                "duration": "1 week",
                "dataset": "Customer transaction data"
            },
            {
                "name": "Sentiment Analysis",
                "description": "Analyze sentiment from product reviews",
                "skills": ["NLP", "NLTK", "Classification"],
                "difficulty": "⭐⭐⭐",
                "duration": "1 week",
                "dataset": "Amazon product reviews"
            }
        ],
        "Advanced Projects": [
            {
                "name": "Image Generation with GANs",
                "description": "Generate realistic images using Generative Adversarial Networks",
                "skills": ["TensorFlow", "Deep Learning", "GANs"],
                "difficulty": "⭐⭐⭐⭐",
                "duration": "2 weeks",
                "dataset": "MNIST/CIFAR-10"
            },
            {
                "name": "Real-time Object Detection",
                "description": "Detect objects in real-time video streams",
                "skills": ["Computer Vision", "YOLO", "OpenCV"],
                "difficulty": "⭐⭐⭐⭐",
                "duration": "2 weeks",
                "dataset": "COCO dataset"
            }
        ]
    }
    
    if project_category in projects:
        for project in projects[project_category]:
            with st.container():
                st.markdown(f"### {project['name']}")
                st.markdown(f"**Description:** {project['description']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Difficulty:** {project['difficulty']}")
                with col2:
                    st.markdown(f"**Duration:** {project['duration']}")
                with col3:
                    st.markdown(f"**Dataset:** {project['dataset']}")
                
                st.markdown("**Skills you'll learn:**")
                for skill in project['skills']:
                    st.markdown(f"- {skill}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"Start Project", key=f"start_{project['name']}"):
                        st.session_state.user_progress['projects_completed'].append(project['name'])
                        st.success("Project started! Check your learning resources.")
                with col2:
                    if st.button(f"View Solution", key=f"solution_{project['name']}"):
                        st.code("""
# Sample solution structure
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv('data.csv')

# Preprocessing
# ... your code here

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
                        """)
                with col3:
                    if st.button(f"Download Dataset", key=f"data_{project['name']}"):
                        st.info("Dataset downloaded! (Simulation)")
                
                st.markdown("---")

elif menu == "Quizzes":
    st.markdown("<h1 class='main-header'>📝 Knowledge Assessment</h1>", unsafe_allow_html=True)
    
    quiz_topic = st.selectbox(
        "Select Quiz Topic",
        ["Python Fundamentals", "Machine Learning", "Deep Learning", "Statistics", "SQL"]
    )
    
    if st.button("Start Quiz"):
        st.session_state.current_quiz = generate_quiz(quiz_topic, 5)
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
    
    if st.session_state.get('current_quiz'):
        st.markdown(f"### Quiz: {quiz_topic}")
        
        for i, q in enumerate(st.session_state.current_quiz):
            st.markdown(f"**Question {i+1}:** {q['question']}")
            answer = st.radio(
                "Select your answer:",
                q['options'],
                key=f"q_{i}",
                index=None
            )
            st.session_state.quiz_answers[i] = q['options'].index(answer) if answer else None
        
        if st.button("Submit Quiz"):
            score = 0
            results = []
            for i, q in enumerate(st.session_state.current_quiz):
                user_answer = st.session_state.quiz_answers.get(i)
                correct = user_answer == q['correct'] if user_answer is not None else False
                if correct:
                    score += 1
                results.append({
                    "question": q['question'],
                    "user_answer": q['options'][user_answer] if user_answer is not None else "Not answered",
                    "correct_answer": q['options'][q['correct']],
                    "explanation": q.get('explanation', ''),
                    "is_correct": correct
                })
            
            percentage = (score / len(st.session_state.current_quiz)) * 100
            st.session_state.user_progress['quiz_scores'][quiz_topic] = percentage
            st.session_state.quiz_results = results
            st.session_state.quiz_submitted = True
            
            if percentage >= 80:
                st.success(f"Excellent! You scored {percentage:.0f}%")
            elif percentage >= 60:
                st.warning(f"Good job! You scored {percentage:.0f}%")
            else:
                st.error(f"Keep practicing! You scored {percentage:.0f}%")
            
        if st.session_state.get('quiz_submitted', False):
            st.markdown("### Detailed Results:")
            for i, result in enumerate(st.session_state.quiz_results):
                with st.expander(f"Question {i+1}"):
                    st.markdown(f"**Your answer:** {'✅' if result['is_correct'] else '❌'} {result['user_answer']}")
                    st.markdown(f"**Correct answer:** {result['correct_answer']}")
                    st.markdown(f"**Explanation:** {result['explanation']}")

elif menu == "Career Guide":
    st.markdown("<h1 class='main-header'>💼 Career Guidance</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Career Paths", "Skills Roadmap", "Interview Prep"])
    
    with tab1:
        st.markdown("### 🎯 AI/Data Science Career Paths")
        
        careers = {
            "Data Scientist": {
                "salary": "$120,000 - $180,000",
                "skills": "Python, ML, Statistics, Communication",
                "description": "Analyze complex data to help companies make decisions"
            },
            "ML Engineer": {
                "salary": "$130,000 - $200,000",
                "skills": "Python, MLOps, Cloud, Software Engineering",
                "description": "Build and deploy ML models at scale"
            },
            "Data Analyst": {
                "salary": "$70,000 - $110,000",
                "skills": "SQL, Excel, Visualization, Business Acumen",
                "description": "Transform data into actionable insights"
            },
            "AI Research Scientist": {
                "salary": "$150,000 - $300,000",
                "skills": "Deep Learning, Research, Mathematics, Publishing",
                "description": "Push the boundaries of AI technology"
            }
        }
        
        for role, info in careers.items():
            with st.expander(f"👔 {role}"):
                st.markdown(f"**Salary Range:** {info['salary']}")
                st.markdown(f"**Key Skills:** {info['skills']}")
                st.markdown(f"**Description:** {info['description']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"View Learning Path", key=f"path_{role}"):
                        path = calculate_learning_path(skill_level, role)
                        st.markdown("**Recommended Learning Path:**")
                        for i, module in enumerate(path, 1):
                            st.markdown(f"{i}. {module}")
                with col2:
                    if st.button(f"Job Openings", key=f"jobs_{role}"):
                        st.info(f"Searching LinkedIn for {role} positions...")
                        time.sleep(1)
                        st.success(f"Found 25+ {role} positions on LinkedIn!")
    
    with tab2:
        st.markdown("### 🗺️ Skills Roadmap")
        
        skill_timeline = {
            "Month 1-2": ["Python Basics", "Git/GitHub", "SQL Fundamentals"],
            "Month 3-4": ["Data Analysis", "Statistics", "Visualization"],
            "Month 5-6": ["Machine Learning", "Feature Engineering", "Model Evaluation"],
            "Month 7-9": ["Deep Learning", "NLP/Computer Vision", "Cloud Platforms"],
            "Month 10-12": ["MLOps", "Production Systems", "Advanced Topics"]
        }
        
        for period, skills in skill_timeline.items():
            st.markdown(f"**{period}:**")
            for skill in skills:
                st.markdown(f"- {skill}")
        
        st.markdown("---")
        st.markdown("### 📈 Skill Demand Analysis")
        skill_demand = {
            "Skill": ["Python", "SQL", "Machine Learning", "Deep Learning", "Cloud", "Data Visualization"],
            "Demand (%)": [95, 85, 90, 75, 80, 70]
        }
        df_demand = pd.DataFrame(skill_demand)
        fig = px.bar(df_demand, x="Skill", y="Demand (%)", 
                     color="Skill", title="Industry Skill Demand")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🎤 Interview Preparation")
        
        interview_topics = {
            "Technical Questions": [
                "Explain the bias-variance tradeoff",
                "What is gradient descent?",
                "Difference between L1 and L2 regularization",
                "How do you handle imbalanced datasets?"
            ],
            "Behavioral Questions": [
                "Tell me about a challenging project",
                "How do you handle conflicting priorities?",
                "Describe a time you worked with stakeholders",
                "How do you stay updated with AI trends?"
            ],
            "Case Studies": [
                "Design a recommendation system",
                "Predict customer churn",
                "Detect fraudulent transactions",
                "Optimize marketing campaigns"
            ]
        }
        
        for category, questions in interview_topics.items():
            with st.expander(f"📚 {category}"):
                for q in questions:
                    st.markdown(f"• {q}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Practice {category}", key=f"practice_{category}"):
                        st.info("Practice session started! Prepare your answers and time yourself.")
                with col2:
                    if st.button(f"View Answers", key=f"answers_{category}"):
                        st.success("Sample answers loaded. Compare with your responses.")

elif menu == "Resume Builder":
    st.markdown("<h1 class='main-header'>📄 ATS-Optimized Resume Builder</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Resume Analysis", "LinkedIn Optimizer", "Cover Letter"])
    
    with tab1:
        st.markdown("### 🔍 ATS Resume Analyzer")
        
        job_role = st.selectbox(
            "Select Target Role",
            list(JOB_TEMPLATES.keys())
        )
        
        resume_text = st.text_area(
            "Paste your resume text here:",
            height=300,
            placeholder="Copy and paste your entire resume content..."
        )
        
        if st.button("Analyze Resume"):
            if resume_text:
                analysis = analyze_resume_ats(resume_text, job_role)
                
                # Display ATS Score
                col1, col2 = st.columns(2)
                with col1:
                    score_color = "green" if analysis['score'] >= 80 else "orange" if analysis['score'] >= 60 else "red"
                    st.markdown(f"### ATS Score: <span style='color:{score_color}'>{analysis['score']:.0f}%</span>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Skills Match", f"{len(analysis['found_skills'])}/{len(JOB_TEMPLATES[job_role]['skills'])}")
                
                # Found skills
                if analysis['found_skills']:
                    st.success("✅ **Skills Found:**")
                    st.write(", ".join(analysis['found_skills']))
                
                # Missing skills
                if analysis['missing_skills']:
                    st.warning("⚠️ **Missing Skills:**")
                    st.write(", ".join(analysis['missing_skills']))
                
                # Recommendations
                st.markdown("### 💡 Recommendations:")
                for rec in analysis['recommendations']:
                    st.markdown(f"• {rec}")
                
                # Save job application
                st.session_state.user_progress['job_applications'].append({
                    "role": job_role,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "score": analysis['score']
                })
            else:
                st.error("Please paste your resume text")
    
    with tab2:
        st.markdown("### 🔗 LinkedIn Profile Optimizer")
        
        linkedin_sections = {
            "Headline": "Data Scientist | Machine Learning | Python | Transforming Data into Insights",
            "Summary": "Passionate data scientist with 3+ years of experience in building ML models that drive business value. Skilled in Python, TensorFlow, and cloud deployment.",
            "Skills": ["Python", "Machine Learning", "Deep Learning", "SQL", "TensorFlow", "PyTorch", "AWS", "Docker"]
        }
        
        for section, content in linkedin_sections.items():
            st.markdown(f"**{section} Template:**")
            if isinstance(content, list):
                st.write(", ".join(content))
            else:
                st.write(content)
        
        st.markdown("### 🎯 LinkedIn Tips:")
        tips = [
            "Use keywords from job descriptions in your headline and summary",
            "Add 50+ skills and get endorsements for top skills",
            "Write detailed descriptions for each role with quantified achievements",
            "Add relevant certifications and courses",
            "Engage with content in your field regularly"
        ]
        
        for tip in tips:
            st.markdown(f"• {tip}")
    
    with tab3:
        st.markdown("### ✉️ Cover Letter Generator")
        
        company_name = st.text_input("Company Name")
        position = st.text_input("Position")
        user_skills = st.multiselect("Your Top Skills", ["Python", "Machine Learning", "Data Analysis", "SQL", "Deep Learning"])
        
        if st.button("Generate Cover Letter"):
            if company_name and position and user_skills:
                cover_letter = generate_cover_letter(company_name, position, user_skills)
                st.text_area("Generated Cover Letter", cover_letter, height=300)
                st.download_button("Download Cover Letter", cover_letter, file_name=f"cover_letter_{company_name}.txt")
            else:
                st.error("Please fill in all fields")

elif menu == "Mind Maps":
    st.markdown("<h1 class='main-header'>🗺️ Visual Learning with Mind Maps</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 🧠 Concept Maps")
        selected_map = st.selectbox("Select a Concept", list(MIND_MAP_DB.keys()))
        
        st.markdown("### ✨ Create Your Own")
        new_map_topic = st.text_input("Map Topic")
        new_concepts = st.text_area("Concepts (comma separated)")
        
        if st.button("Generate Mind Map"):
            if new_map_topic and new_concepts:
                concepts_dict = {concept.strip(): f"Description of {concept.strip()}" 
                                for concept in new_concepts.split(",")}
                # Fixed: Store under user_progress
                st.session_state.user_progress['mind_maps'][new_map_topic] = concepts_dict
                st.success("Mind map created!")
    
    with col2:
        if selected_map in MIND_MAP_DB:
            st.markdown(f"### {selected_map}")
            fig = create_mind_map(selected_map, MIND_MAP_DB[selected_map])
            st.plotly_chart(fig, use_container_width=True)
        
        # Fixed: Access through user_progress
        if st.session_state.user_progress['mind_maps']:
            st.markdown("### 🗂️ Your Custom Maps")
            for topic, concepts in st.session_state.user_progress['mind_maps'].items():
                with st.expander(topic):
                    fig = create_mind_map(topic, concepts)
                    st.plotly_chart(fig, use_container_width=True)

elif menu == "Progress":
    st.markdown("<h1 class='main-header'>📈 Your Learning Progress</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Completed Lessons")
        if st.session_state.user_progress['completed_lessons']:
            for lesson in st.session_state.user_progress['completed_lessons']:
                st.markdown(f"- ✅ {lesson}")
        else:
            st.markdown("No lessons completed yet")
        
        st.markdown("### 🏆 Projects Completed")
        if st.session_state.user_progress['projects_completed']:
            for project in st.session_state.user_progress['projects_completed']:
                st.markdown(f"- 🛠️ {project}")
        else:
            st.markdown("No projects completed yet")
    
    with col2:
        st.markdown("### 📝 Quiz Scores")
        if st.session_state.user_progress['quiz_scores']:
            scores = st.session_state.user_progress['quiz_scores']
            df_scores = pd.DataFrame({
                "Topic": list(scores.keys()),
                "Score": list(scores.values())
            })
            fig = px.bar(df_scores, x="Topic", y="Score", 
                         color="Topic", title="Quiz Performance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("No quiz scores yet")
        
        st.markdown("### 💼 Job Applications")
        if st.session_state.user_progress.get('job_applications'):
            apps = st.session_state.user_progress['job_applications']
            df_apps = pd.DataFrame(apps)
            st.dataframe(df_apps)
        else:
            st.markdown("No job applications tracked yet")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p>AI & Data Science Learning Platform • Built with Streamlit • 
    <a href="https://huggingface.co" target="_blank">Deploy on Hugging Face</a></p>
</div>
""", unsafe_allow_html=True)
