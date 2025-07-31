import time
from user_logger import log_action


class TPSCalculator:
    def __init__(self, tokenizer=None):
        # Initialize tokenizer, raise an error if none is provided or set later
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = (
                None  # Tokenizer needs to be set explicitly if not provided here
            )

        # Initialize variables
        self.input_prompt = None
        self.output_response = None
        self.start_time = None
        self.end_time = None

    def reset(self):
        self.input_prompt = None
        self.output_response = None
        self.start_time = None
        self.end_time = None

    # Set custom tokenizer (useful if not provided during initialization)
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    # Set input prompt
    def set_input_prompt(self, input_prompt):
        self.input_prompt = input_prompt

    # Set output response
    def set_output_response(self, output_response):
        self.output_response = output_response

    # Set start time
    def set_start_time(self):
        self.start_time = time.time()

    # Tokenize text and return token count
    def _get_token_len(self, text):
        if isinstance(text, list):
            # Extract 'content' from each dictionary, ignoring 'role'
            text = " ".join([item["content"] for item in text if isinstance(item, dict) and "content" in item])

        if isinstance(text, str):
            text = text.encode("utf-8")  # Convert string to bytes

        tokens = self.tokenizer.tokenize(text, add_bos=True, special=False)
        return len(tokens)

    # Calculate tokens per second (TPS)
    def calculate_tps(self):
        if self.input_prompt is None or self.output_response is None:
            raise ValueError("Input prompt or output response is not set.")

        if self.start_time is None:
            raise ValueError("Start time is not set.")

        # Get the end_time
        self.end_time = time.time()

        # Get token lengths
        input_token_len = self._get_token_len(self.input_prompt)
        output_token_len = self._get_token_len(self.output_response)

        # Total tokens
        total_tokens_len = input_token_len + output_token_len

        # Elapsed time
        elapsed_time = self.end_time - self.start_time

        if elapsed_time == 0:
            raise ValueError("Elapsed time cannot be zero.")

        # Calculate tokens per second
        tps = output_token_len / elapsed_time

        print(
            "input_tokens:",
            input_token_len,
            " output_tokens:",
            output_token_len,
            " total_tokens:",
            total_tokens_len,
            " elapsed_time:",
            elapsed_time,
            " tps:",
            tps,
            "\n",
        )
        log_action("input_tokens: " + str(input_token_len) +
                " output_tokens: " + str(output_token_len) +
                " total_tokens: " + str(total_tokens_len) +
                " elapsed_time: " + str(elapsed_time) +
                " tps: " + str(tps))
