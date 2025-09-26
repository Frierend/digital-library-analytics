📚 Digital Library Analytics Dashboard

A Streamlit application for exploring borrowing patterns, user behavior, and book recommendations in digital libraries.

🔗 Live Demo (Open in Browser): https://digital-library-analytics-htp3yvnrcxs7d33my8yqqw.streamlit.app/

🎯 Highlights

📊 Dashboard – Borrowing trends, book popularity, ratings, and devices

🔍 Book Search – Analytics, top users, and recommendations for each book

🔗 Association Rules – Book relationships via Apriori/FP-Growth

💡 Automated Insights – Priority-based recommendations for improvement

📱 Device Analysis – Compare desktop, mobile, and tablet usage

📤 Export Reports – Download processed data and insights

🚀 Quick Start
Option 1: Use the Live Demo

Open the Live Demo
.

Upload digital_library_dataset.csv and metadata.csv.

Explore insights instantly.

Option 2: Run Locally
git clone https://github.com/your-username/digital-library-analytics.git
cd digital-library-analytics
pip install -r requirements.txt
streamlit run app.py

📊 Data Requirements

digital_library_dataset.csv → user_id, book_id, borrow_timestamp, return_timestamp, rating, device_type, session_duration, action_type, recommendation_score

metadata.csv → book_id, title, author, year

📈 Example Insights

“Users who borrow Python Programming also borrow Data Science with 85% confidence.”

“Tablet users spend 2x more time per session than mobile users.”

“Peak usage is at 2:00 PM on weekdays.”

📄 License

MIT License – see LICENSE
.

✨ Built with ❤️ using Python & Streamlit