#!/usr/bin/env python3
"""Interactive multi-turn chat for tinyvllm."""

import argparse
import sys

import torch
from transformers import AutoTokenizer

import tinyvllm


class ChatSession:
    """Manages multi-turn conversation state on top of tinyvllm.Engine."""

    def __init__(self, engine, tokenizer, max_new_tokens=512, temperature=0.7):
        self.engine = engine
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.messages = []

    def set_system_prompt(self, content):
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def chat(self, user_message):
        self.messages.append({"role": "user", "content": user_message})

        input_ids = self.tokenizer.apply_chat_template(
            self.messages, tokenize=True, add_generation_prompt=True,
        )

        # Truncate oldest turns if context would overflow.
        max_pos = self.engine.max_position_embeddings()
        while len(input_ids) + self.max_new_tokens > max_pos:
            if not self._drop_oldest_turn():
                break  # nothing left to drop
            input_ids = self.tokenizer.apply_chat_template(
                self.messages, tokenize=True, add_generation_prompt=True,
            )

        params = tinyvllm.SampleParams(
            self.max_new_tokens,
            temperature=self.temperature,
        )
        output_ids = self.engine.generate([input_ids], [params], 0)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        self.messages.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        system = None
        if self.messages and self.messages[0]["role"] == "system":
            system = self.messages[0]
        self.messages = [system] if system else []

    def _drop_oldest_turn(self):
        """Remove the oldest user/assistant pair (preserve system prompt)."""
        start = 1 if self.messages and self.messages[0]["role"] == "system" else 0
        # Need at least the current user message to remain.
        if start + 2 > len(self.messages):
            return False
        del self.messages[start:start + 2]
        return True


def main():
    parser = argparse.ArgumentParser(description="tinyvllm interactive chat")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--hf_model", required=True,
                        help="HuggingFace model name/path for tokenizer")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--system_prompt", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    state_dict = torch.load(
        f"{args.model_dir}/state_dict.pt", map_location="cpu", weights_only=True,
    )
    engine = tinyvllm.Engine(args.model_dir, state_dict)

    session = ChatSession(
        engine, tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    if args.system_prompt:
        session.set_system_prompt(args.system_prompt)

    print(f"tinyvllm chat | max_new_tokens={args.max_new_tokens} "
          f"| temperature={args.temperature}")
    print("Type /help for commands, Ctrl+D to exit.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye!")
            break
        if user_input == "/clear":
            session.reset()
            print("Conversation cleared.\n")
            continue
        if user_input.startswith("/system "):
            session.set_system_prompt(user_input[8:])
            print("System prompt set.\n")
            continue
        if user_input == "/history":
            for msg in session.messages:
                print(f"[{msg['role']}] {msg['content']}")
            print()
            continue
        if user_input == "/help":
            print("/clear   - Reset conversation history")
            print("/system  - Set system prompt")
            print("/history - Show conversation history")
            print("/exit    - Quit")
            print()
            continue

        response = session.chat(user_input)
        print(response)
        print()


if __name__ == "__main__":
    main()
