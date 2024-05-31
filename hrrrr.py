import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import random

# Load leave policies data
leave_policies = pd.read_csv('D:/leave_policies.csv')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

# Initialize and train pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', GradientBoostingClassifier())
])

X = leave_policies['policy'].apply(preprocess_text)
y = leave_policies['response']

pipeline.fit(X, y)

# Chatbot response function
def chatbot_response(user_input):
    user_input = preprocess_text(user_input)
    response = pipeline.predict([user_input])[0]
    return format_response(response)

# Format response function
def format_response(response):
    # Example formatting: Add new lines after periods for readability
    formatted_response = re.sub(r'\.\s+', '.\n', response)
    return formatted_response

# Reinforcement learning feedback loop
feedback_data = []
def collect_feedback(user_input, response, correct_response):
    feedback_data.append((user_input, response, correct_response))
    if len(feedback_data) >= 100:  # Retrain model every 100 feedback instances
        retrain_model()

def retrain_model():
    global pipeline
    X_feedback = [preprocess_text(user_input) for user_input, _, _ in feedback_data]
    y_feedback = [correct_response for _, _, correct_response in feedback_data]
    pipeline.fit(X_feedback, y_feedback)
    feedback_data.clear()

class HRBotGUI:
    def __init__(self, master):
        self.master = master
        master.title("HR Bot")

        # Header Frame
        self.header = tk.Frame(master, bg='#2C3E50', pady=10)
        self.header.pack(fill=tk.X)

        # Load and display the logo with error handling
        try:
            self.logo = Image.open('D:/absax_logo.png')
            self.logo = self.logo.resize((100, 100), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(self.logo)
        except FileNotFoundError:
            self.logo = Image.new('RGB', (100, 100), color='gray')
            self.logo = ImageTk.PhotoImage(self.logo)

        self.logo_label = tk.Label(self.header, image=self.logo, bg='#2C3E50')
        self.logo_label.pack(side=tk.LEFT, padx=10)
        
        self.header_label = tk.Label(self.header, text="Welcome to the ABSAX Technologies Chatbot!", 
                                     fg='white', bg='#2C3E50', font=('Helvetica', 16, 'bold'))
        self.header_label.pack(side=tk.LEFT)

        # Marquee Frame
        self.marquee_frame = tk.Frame(master, bg='#2C3E50')
        self.marquee_frame.pack(fill=tk.X)
        
        self.marquee = tk.Label(self.marquee_frame, text="Welcome to ABSAX Technologies!", 
                                font=('Helvetica', 12, 'bold'), fg='white', bg='#2C3E50')
        self.marquee.pack(fill=tk.X)
        self.marquee_x_pos = master.winfo_width()
        self.update_marquee()

        # Main Frame
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(pady=10)

        self.textbox = tk.Entry(self.main_frame, font=('Helvetica', 14), width=50)
        self.textbox.bind('<Return>', self.ask)
        self.textbox.grid(row=0, column=0, padx=10)

        self.ask_button = tk.Button(self.main_frame, text="Ask", command=self.ask, font=('Helvetica', 14))
        self.ask_button.grid(row=0, column=1, padx=10)

        # Quick Replies Frame
        self.quick_replies_frame = tk.Frame(master)
        self.quick_replies_frame.pack(pady=10)

        common_questions = ["Leave policy", "Notice period policy", "Approval and request"]
        for question in common_questions:
            button = tk.Button(self.quick_replies_frame, text=question, command=lambda q=question: self.quick_reply(q), font=('Helvetica', 12))
            button.pack(side=tk.LEFT, padx=5)

        # Output Frame
        self.output_frame = tk.Frame(master)
        self.output_frame.pack(pady=10)

        self.output = scrolledtext.ScrolledText(self.output_frame, height=20, width=80, font=('Helvetica', 14))
        self.output.pack()
        self.output.tag_config('bot', foreground='blue',font=('Helvetica', 14, 'bold'))

        self.quit_button = tk.Button(master, text="Quit", command=master.quit, font=('Helvetica', 14), bg='#E74C3C', fg='white')
        self.quit_button.pack(pady=10)

        # Welcome message
        self.output.insert(tk.END, "HR Bot: Hello! How can I assist you today?\n", 'bot')

        # Bottom Frame
        self.bottom_frame = tk.Frame(master, bg='#2C3E50', pady=10)
        self.bottom_frame.pack(fill=tk.X)

        self.status_label = tk.Label(self.bottom_frame, text="Chatbot powered by ABSAX Technologies", 
                                     font=('Helvetica', 10, 'bold'), fg='white', bg='#2C3E50')
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.clear_button = tk.Button(self.bottom_frame, text="Clear", 
                                      command=self.clear_messages)
        self.clear_button.pack(side=tk.RIGHT)

        self.feedback_button = tk.Button(self.bottom_frame, text="Feedback", 
                                        command=self.collect_feedback)
        self.feedback_button.pack(side=tk.RIGHT)

        self.output.see(tk.END)

    def ask(self, event=None):
        user_input = self.textbox.get()
        if user_input.strip() == "":
            return
        
        if user_input.lower() in ['quit', 'bye', 'exit']:
            #self.output.insert(tk.END, "HR Bot: Goodbye!\n", 'bot')
            #self.master.quit()
            self.master.destroy()
        else:
            response = chatbot_response(user_input)
            self.output.insert(tk.END, f"You: {user_input}\n")
            self.output.insert(tk.END, f"HR Bot: {response}\n", 'bot')
            self.textbox.delete(0, tk.END)
            self.output.see(tk.END)

    def quick_reply(self, question):
        self.textbox.delete(0, tk.END)
        self.textbox.insert(0, question)
        self.ask()

    def clear_messages(self):
        self.output.delete(1.0, tk.END)

    def collect_feedback(self, event=None):
        def submit_feedback():
            user_input = self.textbox.get()
            response = chatbot_response(user_input)
            correct_response = feedback_entry.get()
            collect_feedback(user_input, response, correct_response)
            feedback_window.destroy()
            
        if event is not None and event.keysym == "Return":  # Check if Enter key is pressed
            submit_feedback()
            return
        
        feedback_window = tk.Toplevel(self.master)
        feedback_window.title("Feedback")

        tk.Label(feedback_window, text="Please enter the correct response:").pack()
        feedback_entry = tk.Entry(feedback_window, font=('Helvetica', 14), width=50)
        feedback_entry.pack()

        submit_button = tk.Button(feedback_window, text="Submit", command=submit_feedback)
        submit_button.pack()
    
    def update_marquee(self):
        text = self.marquee.cget("text")
        if len(text) > 0:
            self.marquee.config(text=text[1:] + text[0])
            self.marquee.after(100, self.update_marquee)


def open_chatbot():
    avatar_button.pack_forget()  # Hide the avatar button
    chatbot_gui = HRBotGUI(root)
    chatbot_gui.pack(fill=tk.BOTH, expand=True)

def main():
    global root
    root = tk.Tk()
    root.title("Main Window")

    # Load and display the avatar icon
    try:
        avatar = Image.open('D:/chatbot_avataar.png')
        avatar = avatar.resize((100, 100), Image.LANCZOS)
        avatar = ImageTk.PhotoImage(avatar)
    except FileNotFoundError:
        avatar = Image.new('RGB', (100, 100), color='gray')
        avatar = ImageTk.PhotoImage(avatar)

    global avatar_button
    avatar_button = tk.Button(root, image=avatar, command=open_chatbot)
    avatar_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()