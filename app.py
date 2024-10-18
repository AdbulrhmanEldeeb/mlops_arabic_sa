import gradio as gr
import joblib

# Load the pre-trained model and vectorizer
model = joblib.load('models/stacking_model.pkl')
vectorizer = joblib.load('models/count_vectorizer.pkl')

# Define the prediction function
def predict_class(text):
    text_vector = vectorizer.transform([text])
    class_value = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)
    class_name = 'ايجابي' if class_value == 1 else 'سلبي'
    return class_name, round(probability[0][class_value], 2)

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_class,
    inputs="textbox",  # Input is a textbox where users input their text
    outputs=[gr.Textbox(label="التصنيف"), gr.Textbox(label="النسبة")],  # Output labels for class and probability
    title="تصنيف المشاعر",
    description="أدخل النص لتحديد المشاعر (إيجابي أو سلبي) والنسبة."
)

# Launch the Gradio app
interface.launch()
