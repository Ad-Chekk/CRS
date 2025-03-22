# 🎓 Course Recommendation System

  Overview
The **Course Recommendation System** is a **machine learning-powered web app** built with **Streamlit** that suggests courses based on similarity of course embeddings. It leverages **BERT-based embeddings** and **cosine similarity** to find the best matching courses.

 Features
- 🔍 **Course Selection**: Choose a course from the dropdown, and get top 5 recommended courses based on similarity.
- 📌 **Recommendation Cards**: Displayed in **aesthetic semi-transparent cards** with course details.
- 🏫 **University & Ratings**: Each recommendation includes university name and course rating.
- 🌐 **Course Search**: Search for courses by name.
- 💾 **Processed Data Export**: Save the processed courses as a CSV file.

 Screenshots
![image](https://github.com/user-attachments/assets/20f6b43e-6f89-4154-8588-708ae708768b)


  Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/course-recommendation.git
cd course-recommendation
```
 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
 3️⃣ Run the Application
```bash
streamlit run app.py
```

📂 Project Structure
```
📁 course-recommendation/
├── 📄 app.py            # Streamlit UI & Recommendation Logic
├── 📄 course_embeddings.pkl # Precomputed Course Embeddings
├── 📄 processed_embeddings.csv # Expanded Embeddings Data
├── 📄 processed_courses.csv # Processed Course Data
├── 📄 requirements.txt  # Dependencies
└── 📄 README.md         # Project Documentation
```

 How It Works
1. **Loads Precomputed Embeddings**: Reads `course_embeddings.pkl` (created using BERT model).
2. **Processes Embeddings**: Converts tensors to lists and expands them into individual dimensions.
3. **Computes Similarity**: Uses **cosine similarity** to find the closest matching courses.
4. **Displays Recommendations**: Top 5 courses are shown in beautifully styled **semi-transparent cards**.

 🤖 Technologies Used
- **Python** 🐍
- **Streamlit** 🎨 (For UI)
- **Torch (PyTorch)** 🔥 (For Tensor Handling)
- **scikit-learn** 🧠 (For Similarity Computation)
- **pandas** 🏗 (For Data Processing)

📌 Future Enhancements
- ✨ **User-based personalization** (Store user preferences for better recommendations)
- 🔥 **More advanced NLP models** (Try sentence transformers for better embeddings)
- 🌍 **Deploy on a cloud platform** (Streamlit Cloud / AWS / Heroku)

## 💡 Contributing
Contributions are welcome! Feel free to **fork** the repo and submit a **pull request**.

## 📜 License
This project is licensed under the MIT License.

---


