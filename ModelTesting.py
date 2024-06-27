import os
import json
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login


# Load configuration from config.json
with open('token.json') as config_file:
    config = json.load(config_file)

HFT = config['HF_TOKEN']
login(HFT)

#@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", use_auth_token=HFT)
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", use_auth_token=HFT)
    return tokenizer, model

# Main app function
def main():
    st.title("Machine Learning Definition Generator")

    # Input field for user prompt
    user_prompt = st.text_input("Enter your prompt (e.g., Write me a definition on Natural Language Processing.)")

    # Generate button
    if st.button("Generate Definition"):
        if user_prompt:
            tokenizer, model = load_model()  # Load model and tokenizer only when needed

            input_ids = tokenizer(user_prompt, return_tensors="pt")
            outputs = model.generate(**input_ids, max_length=300)
            definition = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.success("Here's the generated definition:")
            st.write(definition)
        else:
            st.warning("Please enter a prompt to generate a definition.")

if __name__ == "__main__":
    main()
