# AI Backend

<p align="center">
    <em>A simple interface to easily swap between AI providers and specific models.</em>
</p>

[![build](https://github.com/ai_backend/workflows/Build/badge.svg)](https://github.com/ai_backend/actions)
[![codecov](https://codecov.io/github/jcmullwh/ai_backend/branch/main/graph/badge.svg?token=ZOE3PNF04X)](https://codecov.io/github/jcmullwh/ai_backend)
[![PyPI version](https://badge.fury.io/py/ai_backend.svg)](https://badge.fury.io/py/ai_backend)

---

**Documentation**: <a href="https://jcmullwh.github.io/ai_backend/" target="_blank">https://jcmullwh.github.io/ai_backend/</a>

**Source Code**: <a href="https://github.com/ai_backend" target="_blank">https://github.com/ai_backend</a>

---

### Purpose:

The purpose of AI Backend is to provide a generic API for accessing AI models that is intuitive and easy to use, build on, maintain, and extend. 
Complete abstraction when desired combined with fine-grain control when needed.

Kind of like a slice of LangChain but not absolutely terrible. 

---

### Examples

```python

text_ai = TextAI(backend="openai")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "This is a test message. Please respond with 'Test response'."},
]

default_model_response = text_ai.text_chat(messages)

gpt_4o_response = text_ai.text_chat(messages, model="gpt-4o")

text_ai.set_default("text_chat",model="gpt-4o")

also_gpt_4o_response = text_ai.text_chat(messages)

# At any point, parameters can be passed as kwargs.
# If it's a call to the model it will use these parameters.

response_with_a_bunch_of_parameters = text_ai.text_chat(messages,
                                            frequency_penalty=1.1,
                                            max_tokens=42,
                                            presence_penalty=1.2,
                                            n=4,
                                            )

# Parameters can be set as default at initialization and using set_default()

new_text_ai = TextAI(frequency_penalty=1.1)

response_with_modified_frequency_penalty = new_text_ai(messages)

new_text_ai.set_default(frequency_penalty=1)

response_with_frequency_penalty_1 = new_text_ai(messages)


# Future:

text_ai.set_backend("google")

default_google_response = text_ai.text_chat(messages)

```

---

### Status

Basic Functionality:
- [x] OpenAI Backend
- [x] Capability API
- [x] Text
- [x] Image
- [ ] Audio
- [ ] 90% Test Coverage
- [ ] Clear Logging of all changes to model parameters
- [ ] Message/Memory Handling

Addtional Backends:

- [ ] Google
- [ ] Anthropic
- [ ] Meta
- [ ] Stability
- [ ] Midjourney

Additional Capabilities:
- [ ] Functions
- [ ] Embeddings



