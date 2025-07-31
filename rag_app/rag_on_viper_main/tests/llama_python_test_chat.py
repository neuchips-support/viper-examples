from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(
    model_path="/data/gguf/neuchips/llama2/models/llama-2-7b-chat/ggml-model-f16.gguf",
    chat_format="llama-2",
    verbose=True,  # Enable verbose logging for debugging
    n3k_id=0,
    neutorch_cache_path="./data/precompiler_cache/meta-llama/Llama-2-7b-chat-hf/gguf/",
)

def chat():
    print("Welcome to Llama Chat! Type 'exit' to end the chat.")

    while True:
        # Get input from the user
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break

        try:
            print("Llama: ", end="", flush=True)
            stream_response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant, aiming to keep your answers in brief."},
                    {"role": "user", "content": user_input},
                ],
                stream=True
            )

            assistant_response = ""
            for chunk in stream_response:
                token = chunk["choices"][0]["delta"].get("content", "")
                print(token, end="", flush=True)
                assistant_response += token

            print("\n")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    chat()
