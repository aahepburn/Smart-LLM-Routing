


# Structure of the repo

router/ (core routing logic)
models/ (classifier + responsive evaluator)
training/ (data prep + training loops)
api/ (FastAPI)
demo/ (notebook)


# Models to be used

Weak model: Phi-2

Strong model: Mistral

Embeddings: text-embedding-3-large

Evaluator: BERT-base


# Response Quality Evaluator

- Inputs:
    - Prompt text
    - Weak model response text
    - target/reference answer when training, or a quality label

- Architecture:
    - Concatenate prompt + response with separators
    - Tokenize with BERT
    - Pass through BERT encoder
    - User a custom classification head

- Outputs:
    - Scalar in $$$$ meaning "adequate response probability," or a 2-class softmax output.


# Training Pipeline

1. Data:
    Build a synthetic dataset:
        - Sampling prompts from open datasets
        - Collecting weak and strong responses
        - Labelling:
            - Use a stronger LLM as a judge to score each respone 


2. PyTorch data code:
        - Custom Dataset and Dataloader
    
3. Training loop:
        - forward -> loss --> loss.backward() --> optimizer.step() --> scheduler.step() --> optimizer.zero_grad()

# Router + Integration

1. Prompt router:
    - Optionally re-use a simple difficulty classifier (BERT based) to decide whether to even try the weak model

2. Flow:

    - User prompt -> router decides to try the weak model.
    - Call weak model -> get response.
    - Run evaluator (in torch.no_grad()) to score quality
    - If evaluation is inadequate:
        - call strong model
        - return strong model response
    else:
        - return weak model response

3. Glue code:
    - Implement as a FastAPI endpoint: POST /chat with {prompt: ...} and JSON response containing chosen model, score, and text.

# Add Evaluation and "Matrix Factorisation" Scoring

- Create model embeddings v_m (small vector per model) and query embeddings v_q (from an encoder or OpenAI embeddings)

- Define a simple scoring function in PyTorch that uses:
    - Hadamard product between v_m and a transformed v_q
    - A final linear layer to output a scalar

- Use this to:
    - Pre-route "easy" queries to the weak model
    - Compare with the evaluator-based strategy


8. Practice the Interview Questions in Context
While building, explicitly prepare answers to their five questions using your project:
	1.	 loss.backward()  and computation graphs:
	•	Write a short markdown note explaining gradients, autograd, and why  torch.no_grad()  is used in your evaluator during inference.
	2.	Matrix factorization router:
	•	Derive and implement a toy version of their scoring function in your code, then explain in comments what it computes and why it’s called matrix factorization.
	3.	Response quality evaluator:
	•	Describe exactly what you built (inputs/outputs/architecture/data).
	4.	Custom PyTorch project:
	•	Use this router as your answer, with details on the loss function and why you chose it.
	5.	“Why should we hire you?”:
	•	Anchor your answer in the fact that you already built a very similar system end‑to‑end.

    
9. Concrete Timeline (4–5 Focused Days)
If you timebox:
	•	Day 1: Design doc, choose models/APIs, outline repo, write data schema.
	•	Day 2: Implement Dataset/DataLoader, evaluator model, and training loop; generate small synthetic dataset.
	•	Day 3: Train evaluator, implement router logic, build a basic FastAPI endpoint.
	•	Day 4: Add evaluation scripts, documentation, tests, and a demo notebook or UI.
	•	Day 5 (optional): Refine, clean code, record demo video, and integrate into your portfolio/Upwork profile.
If you’d like, next step I can help you sketch the repo structure and exact class/function signatures so you can start coding without decision fatigue.