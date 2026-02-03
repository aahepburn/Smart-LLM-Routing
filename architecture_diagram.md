# Smart LLM Routing Service - Architecture Diagram

```mermaid
graph TB
    subgraph "API Layer"
        FastAPI[FastAPI Application<br/>api_server/main.py]
        
        Health["/health<br/>health_check()"]
        Models["/v1/models<br/>list_models()"]
        Completion["/v1/chat/completions<br/>chat_completion()"]
        
        FastAPI --> Health
        FastAPI --> Models
        FastAPI --> Completion
    end
    
    subgraph "Schemas"
        Schemas[api_server/schemas.py]
        ChatMessage["ChatMessage<br/>- role: str<br/>- content: str"]
        ChatRequest["ChatCompletionRequest<br/>- model: str<br/>- messages: list<br/>- temperature: float<br/>- force_model: Optional"]
        ChatResponse["ChatCompletionResponse<br/>- id: str<br/>- choices: list<br/>- usage: UsageInfo<br/>- routing_metadata: RoutingMetadata"]
        
        Schemas --> ChatMessage
        Schemas --> ChatRequest
        Schemas --> ChatResponse
    end
    
    subgraph "Pipeline Orchestration"
        Pipeline[router_service/pipeline.py]
        AnswerQuestion["answer_question()<br/>Main Pipeline Function"]
        
        Pipeline --> AnswerQuestion
        
        PipelineSteps["Pipeline Flow:<br/>1. Route to weak/strong model<br/>2. Get response from chosen model<br/>3. Evaluate weak response quality<br/>4. Re-route if quality low<br/>5. Return response + metadata"]
        
        AnswerQuestion -.-> PipelineSteps
    end
    
    subgraph "Router Service"
        Router[router_service/router.py]
        RouterService["RouterService<br/>Main Routing Logic"]
        
        InitController["_initialize_controller()<br/>- Setup RouteLLM Controller<br/>- Configure strong/weak models"]
        RoutePrompt["route_prompt()<br/>- Decide model selection<br/>- Execute LLM API call<br/>Returns: (model_name, response)"]
        RouteOnly["route_only()<br/>- Get routing decision only<br/>Returns: model_name"]
        
        Router --> RouterService
        RouterService --> InitController
        RouterService --> RoutePrompt
        RouterService --> RouteOnly
        
        RouteLLMController["RouteLLM Controller<br/>External Library"]
        InitController -.-> RouteLLMController
    end
    
    subgraph "Evaluator Model"
        EvalModels[evaluator_model/models.py]
        
        ResponseQualityModel["ResponseQualityModel<br/>Neural Network"]
        RQMethods["Methods:<br/>- encode_pair()<br/>- forward()<br/>- predict_quality()<br/>- from_pretrained()"]
        
        QualityEvaluator["QualityEvaluator<br/>Wrapper Class"]
        QEMethods["Methods:<br/>- evaluate()<br/>- evaluate_batch()"]
        
        EvalModels --> ResponseQualityModel
        EvalModels --> QualityEvaluator
        ResponseQualityModel --> RQMethods
        QualityEvaluator --> QEMethods
        
        Encoder["Transformer Encoder<br/>BERT/RoBERTa/DeBERTa"]
        Classifier["Classification Head<br/>Hidden → Hidden/2 → 1<br/>Sigmoid Output [0,1]"]
        
        ResponseQualityModel -.-> Encoder
        ResponseQualityModel -.-> Classifier
    end
    
    subgraph "Inference"
        Infer[evaluator_model/infer.py]
        EvaluatorWrapper["Evaluator<br/>Convenience Wrapper"]
        InferMethods["Methods:<br/>- evaluate()<br/>- evaluate_batch()"]
        
        Infer --> EvaluatorWrapper
        EvaluatorWrapper --> InferMethods
    end
    
    subgraph "Training"
        Train[evaluator_model/train.py]
        
        TrainEpoch["train_epoch()<br/>- Forward pass<br/>- Compute loss<br/>- Backward pass<br/>- Update weights"]
        ValidateEpoch["validate_epoch()<br/>- Evaluation mode<br/>- Compute metrics<br/>- No gradient updates"]
        TrainModel["train_model()<br/>Main Training Loop"]
        
        Train --> TrainEpoch
        Train --> ValidateEpoch
        Train --> TrainModel
    end
    
    subgraph "Data Pipeline"
        DataPipeline[data_pipeline/]
        
        GenPrompts["generate_prompts.py<br/>- generate_synthetic_prompts()<br/>- load_prompts_from_file()"]
        CollectResponses["collect_responses.py<br/>- collect_responses()<br/>- query_llm()"]
        BuildDataset["build_evaluator_dataset.py<br/>- create_training_dataset()<br/>- create_preference_pairs()"]
        
        DataPipeline --> GenPrompts
        DataPipeline --> CollectResponses
        DataPipeline --> BuildDataset
    end
    
    subgraph "Data Layer"
        EvalData[evaluator_model/data.py]
        
        ResponsePairDataset["ResponsePairDataset<br/>PyTorch Dataset"]
        DataMethods["Methods:<br/>- __getitem__()<br/>- __len__()<br/>- collate_fn()"]
        
        EvalData --> ResponsePairDataset
        ResponsePairDataset --> DataMethods
    end
    
    subgraph "External Systems"
        OpenAI["OpenAI API<br/>GPT-4 / GPT-3.5"]
        Anthropic["Anthropic API<br/>Claude Models"]
        OtherLLMs["Other LLM APIs"]
    end
    
    %% Flow connections
    Completion --> AnswerQuestion
    Schemas -.-> Completion
    Schemas -.-> AnswerQuestion
    
    AnswerQuestion --> RouterService
    AnswerQuestion --> EvaluatorWrapper
    
    RoutePrompt --> OpenAI
    RoutePrompt --> Anthropic
    RoutePrompt --> OtherLLMs
    
    EvaluatorWrapper --> ResponseQualityModel
    
    TrainModel --> ResponseQualityModel
    TrainModel --> ResponsePairDataset
    
    GenPrompts --> CollectResponses
    CollectResponses --> BuildDataset
    BuildDataset --> ResponsePairDataset
    
    CollectResponses --> OpenAI
    CollectResponses --> Anthropic
    
    %% Styling
    classDef apiClass fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef routerClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef evalClass fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef dataClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef externalClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class FastAPI,Health,Models,Completion,Schemas apiClass
    class Router,RouterService,Pipeline,AnswerQuestion routerClass
    class EvalModels,ResponseQualityModel,QualityEvaluator,Infer,EvaluatorWrapper,Train evalClass
    class DataPipeline,EvalData,ResponsePairDataset dataClass
    class OpenAI,Anthropic,OtherLLMs externalClass
```

## Component Overview

### 1. **API Layer** (`api_server/`)
- **FastAPI Application**: Main entry point for REST API
- **Endpoints**:
  - `GET /health`: Health check
  - `GET /v1/models`: List available models (OpenAI compatible)
  - `POST /v1/chat/completions`: Main chat completion endpoint
- **Schemas**: Pydantic models for request/response validation

### 2. **Router Service** (`router_service/`)
- **RouterService**: Core routing logic using RouteLLM
  - Integrates with RouteLLM Controller for intelligent routing
  - Manages strong model (e.g., GPT-4) and weak model (e.g., GPT-3.5)
  - Methods: `route_prompt()`, `route_only()`, `_initialize_controller()`

- **Pipeline**: Orchestration layer
  - `answer_question()`: Main pipeline function
  - Flow: Route → Get Response → Evaluate → Re-route if needed

### 3. **Evaluator Model** (`evaluator_model/`)
- **ResponseQualityModel**: Neural network for quality evaluation
  - Architecture: Transformer Encoder (BERT/DeBERTa) + Classification Head
  - Input: `[CLS] prompt [SEP] response [SEP]`
  - Output: Quality score in [0, 1]
  - Methods: `encode_pair()`, `forward()`, `predict_quality()`

- **QualityEvaluator**: High-level wrapper for model
  - Methods: `evaluate()`, `evaluate_batch()`

- **Evaluator (infer.py)**: Convenience class for inference
  - Loads checkpoint and manages model lifecycle

- **Training** (`train.py`):
  - `train_epoch()`: Training loop with backpropagation
  - `validate_epoch()`: Validation without gradient updates
  - `train_model()`: Main training orchestration

### 4. **Data Pipeline** (`data_pipeline/`)
- **Generate Prompts**: Create or load test prompts
- **Collect Responses**: Query LLMs and gather responses
- **Build Dataset**: Create training data with preference pairs

### 5. **Data Layer** (`evaluator_model/data.py`)
- **ResponsePairDataset**: PyTorch Dataset implementation
- Handles loading and preprocessing of training data

### 6. **External Systems**
- OpenAI API (GPT-4, GPT-3.5-turbo)
- Anthropic API (Claude models)
- Other LLM providers

## Request Flow

```
User Request
    ↓
FastAPI Endpoint (/v1/chat/completions)
    ↓
answer_question() [Pipeline]
    ↓
RouterService.route_prompt()
    ↓
RouteLLM Controller Decision
    ↓
LLM API Call (OpenAI/Anthropic)
    ↓
If Weak Model Response:
    ↓
    Evaluator.evaluate(prompt, response)
    ↓
    If Quality < Threshold:
        ↓
        Re-route to Strong Model
    ↓
Return Response + Metadata
```

## Key Design Patterns

1. **Layered Architecture**: API → Pipeline → Router → LLM
2. **Strategy Pattern**: Router decides model based on prompt characteristics
3. **Decorator Pattern**: Evaluator wraps responses with quality scores
4. **Factory Pattern**: Model loading via `from_pretrained()`
5. **Pipeline Pattern**: Sequential processing with conditional branches

## Configuration

- Models configured via `RouterService` initialization
- Quality threshold configurable in `answer_question()`
- Encoder model selectable in `ResponseQualityModel`
- Training hyperparameters in training scripts
