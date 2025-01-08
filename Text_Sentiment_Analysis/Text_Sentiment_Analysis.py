from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.config import Config

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

# Global variables for model and tokenizer
model = None
tokenizer = None

# Labels for sentiment
labels = [
    "Negative",
    'Neutral',
    'Positive'
]

# Flag to check if model is loaded
is_model_loaded = False

# Interface--------------------------------------------------------------------------
Config.set('graphics', 'width', '950')
Config.set('graphics', 'height', '650')
Config.set('graphics', 'resizable', False)

class Interface(App):
    def build(self):
        layout = FloatLayout()

        # Head Text----------------------------------------
        label = Label(
            pos=(0, 150),
            text_size=(350, 200),
            halign="center",
            valign="top",
            font_size=24,
            text="Please enter the text for sentiment analysis", 
        )   

        # Text Input----------------------------------------
        self.text_input = TextInput(
            size_hint=(None, None),
            size=(435, 120),
            pos_hint={"center_x": 0.5, "center_y": 0.65},
            hint_text="Enter text here (up to 600 characters)",
            multiline=True,
        )

        # Character Limit
        def limit_characters(instance, value):
            if len(value) > 600:
                instance.text = value[:600]

        self.text_input.bind(text=limit_characters)

        # Button----------------------------------------
        button = Button(
            text="Analyze",
            size_hint=(None, None),
            size=(150, 50),
            pos_hint={"center_x": 0.5, "center_y": 0.45},
        )

        # Sentiment Analysis----------------------------------------
        def on_button_press(instance):
            # Show the loading message
            loading_label.opacity = 1
            # Hide the sentiment labels
            self.negative_label.opacity = 0
            self.neutral_label.opacity = 0
            self.positive_label.opacity = 0

            print(f"Text entered: {self.text_input.text}")
            # preprocess text
            text = self.text_input.text
            text_words = []

            for word in text.split(' '):
                if word.startswith('@') and len(word) > 1:
                    word = '@user'

                elif word.startswith('http'):
                    word = 'http'
                text_words.append(word)

            text_proc = " ".join(text_words)

            # sentiment analysis
            encoded_text = tokenizer(text_proc, return_tensors='pt')
            output = model(**encoded_text)

            scores = output[0][0].detach().numpy()
            scores = softmax(scores)

            # Display sentiment analysis result
            self.update_sentiment_labels(scores)

            # Hide loading and show the labels
            loading_label.opacity = 0
            self.negative_label.opacity = 1
            self.neutral_label.opacity = 1
            self.positive_label.opacity = 1

        button.bind(on_press=on_button_press)

        # Loading Label----------------------------------------
        loading_label = Label(
            text="Loading...",
            size_hint=(None, None),
            size=(150, 50),
            pos_hint={"center_x": 0.5, "center_y": 0.4},
            opacity=0  # Hidden initially
        )

        # Results----------------------------------------
        self.negative_label = Label(
            text="Negative: 0",
            size_hint=(None, None),
            size=(300, 40),
            pos_hint={"center_x": 0.5, "center_y": 0.25},
            color=(1, 0, 0, 1),  # Red color for Negative
            opacity=0  # Hidden initially
        )

        self.neutral_label = Label(
            text="Neutral: 0",
            size_hint=(None, None),
            size=(300, 40),
            pos_hint={"center_x": 0.5, "center_y": 0.20},
            color=(1, 1, 0, 1),  # Yellow color for Neutral
            opacity=0  # Hidden initially
        )

        self.positive_label = Label(
            text="Positive: 0",
            size_hint=(None, None),
            size=(300, 40),
            pos_hint={"center_x": 0.5, "center_y": 0.15},
            color=(0, 1, 0, 1),  # Green color for Positive
            opacity=0  # Hidden initially
        )

        layout.add_widget(label)
        layout.add_widget(self.text_input)
        layout.add_widget(button)
        layout.add_widget(loading_label)

        # Add sentiment result labels to the layout
        layout.add_widget(self.negative_label)
        layout.add_widget(self.neutral_label)
        layout.add_widget(self.positive_label)

        return layout

    def update_sentiment_labels(self, scores):
        # Update sentiment labels with the new scores
        self.negative_label.text = f"Negative: {scores[0]:.2f}"
        self.neutral_label.text = f"Neutral: {scores[1]:.2f}"
        self.positive_label.text = f"Positive: {scores[2]:.2f}"

# Model loading function
def load_model():
    global model, tokenizer
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    print("Model loaded.")

# Load the model and start application
def load_model_and_start_app():
    load_model()
    app = Interface()
    app.run()

if __name__ == '__main__':
    load_model_and_start_app()
