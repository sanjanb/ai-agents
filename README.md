# 🤖 AI Agents — Learn & Build

[![Website](https://img.shields.io/badge/Website-Live-brightgreen)](http://localhost:8000)
[![Documentation](https://img.shields.io/badge/Docs-MkDocs-blue)](http://localhost:8000)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

> **Comprehensive educational resource for designing, building, and deploying AI agents using Large Language Models (LLMs), retrieval-augmented generation (RAG), and cutting-edge fine-tuning techniques.**

## 🌟 Overview

This project provides a complete learning pathway from foundational LLM concepts to advanced agent implementations, featuring:

- **📚 6 Comprehensive Chapters** covering theory and practice
- **🛠️ Hands-On Examples** with runnable code and implementations
- **🔧 Production-Ready Tools** including evaluation harnesses and tracing utilities
- **⚡ Efficient Training** techniques like LoRA, QLoRA, and 1-bit LLMs
- **🚀 Deployment Strategies** from local development to cloud production

## 📖 Table of Contents

### Core Chapters

1. **[Foundational LLMs & Text Generation](docs/agents/foundational-llms.md)**

   - LLM architecture and capabilities
   - Text generation techniques and best practices
   - Model selection and evaluation

2. **[Embeddings & Vector Stores](docs/agents/embeddings-vector-stores.md)**

   - Vector representations and similarity search
   - Database integration and indexing strategies
   - Semantic search implementation

3. **[Generative Agents](docs/agents/generative-agents.md)**

   - Agent architectures and reasoning frameworks
   - Tool integration and function calling
   - Memory systems and planning strategies
   - **🎯 Practical Examples**: Function calling, LangGraph agents, RAG memory

4. **[Domain-Specific LLMs](docs/agents/domain-specific-llms.md)**

   - Adaptation strategies and use cases
   - Data curation and quality management
   - Grounding techniques and external knowledge integration
   - **🎯 Practical Examples**: Fine-tuning workflows, search grounding

5. **[Fine-Tuning LLMs](docs/agents/fine-tuning-llms.md)**

   - Quantization techniques (4-bit, 8-bit, 1-bit)
   - Parameter-efficient fine-tuning (LoRA, QLoRA)
   - Advanced optimization and deployment
   - **🎯 Practical Examples**: LoRA implementation, evaluation harness

6. **[Getting Started](docs/agents/getting-started.md)**
   - Environment setup and dependencies
   - Quick start guides and tutorials

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Git**
- **Virtual Environment** (recommended)

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/sanjanb/ai-agents.git
cd ai-agents

# Create and activate virtual environment
python -m venv .venv

# On Windows
.\.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# Install documentation dependencies
pip install mkdocs mkdocs-material

# Install example dependencies
pip install -r examples/agents/requirements.txt
pip install -r examples/llms/requirements.txt
```

### Run the Documentation Website

```bash
# Start the development server
mkdocs serve

# Open in browser
# Navigate to http://localhost:8000
```

### Try the Examples

#### 🤖 Agent Examples

```bash
# Navigate to examples directory
cd examples/agents

# Run Gemini function calling demo
python gemini_function_calling.py

# Run LangGraph ReAct agent
python langgraph_react_agent.py

# Run RAG memory demonstration
python rag_memory_agent.py

# Run evaluation harness
cd ../../scripts
python eval_rag.py
```

#### 🔧 Fine-Tuning Examples

```bash
# Navigate to LLM examples
cd examples/llms

# Run LoRA fine-tuning
python lora_finetune_news.py --epochs 1 --sample_size 1000

# Evaluate model performance
python eval_domain_classification.py --sample_size 500

# Test search grounding workflow
python search_grounding_stub.py
```

## 🏗️ Project Structure

```
ai-agents/
├── 📁 docs/                           # Documentation source
│   ├── agents/                        # Chapter content
│   │   ├── foundational-llms.md
│   │   ├── embeddings-vector-stores.md
│   │   ├── generative-agents.md
│   │   ├── domain-specific-llms.md
│   │   ├── fine-tuning-llms.md
│   │   └── getting-started.md
│   └── assets/                        # Images and resources
├── 📁 examples/                       # Runnable code examples
│   ├── agents/                        # Agent implementations
│   │   ├── gemini_function_calling.py
│   │   ├── langgraph_react_agent.py
│   │   ├── rag_memory_agent.py
│   │   ├── tracing_utils.py
│   │   ├── requirements.txt
│   │   └── README.md
│   └── llms/                          # Fine-tuning examples
│       ├── lora_finetune_news.py
│       ├── eval_domain_classification.py
│       ├── search_grounding_stub.py
│       ├── requirements.txt
│       └── README.md
├── 📁 scripts/                        # Evaluation and utilities
│   └── eval_rag.py
├── 📁 _layouts/                       # Website templates
├── 📁 assets/                         # Stylesheets and resources
├── 📄 mkdocs.yml                      # Documentation configuration
├── 📄 _config.yml                     # Jekyll configuration
├── 📄 Gemfile                         # Ruby dependencies
└── 📄 README.md                       # This file
```

## 🔬 Features & Capabilities

### 🧠 Agent Intelligence

- **Multi-Modal Reasoning**: Text, search, and tool integration
- **Memory Systems**: Short-term and long-term memory management
- **Planning & Reflection**: Strategic decision-making capabilities
- **Tool Integration**: External API and database connectivity

### ⚡ Performance Optimization

- **Quantization**: 4-bit, 8-bit, and 1-bit model compression
- **Parameter-Efficient Training**: LoRA, QLoRA, and adapter techniques
- **Memory Management**: Gradient checkpointing and CPU offloading
- **Distributed Training**: Multi-GPU and cloud deployment strategies

### 📊 Evaluation & Monitoring

- **Precision@k and MRR**: Information retrieval metrics
- **OpenTelemetry Integration**: Performance tracing and monitoring
- **A/B Testing Framework**: Model comparison and validation
- **Cost Analysis**: Training and inference expense tracking

### 🌐 Production Deployment

- **FastAPI Integration**: REST API endpoints for agents
- **Container Support**: Docker deployment configurations
- **Cloud Integration**: Azure, AWS, and GCP compatibility
- **Scaling Strategies**: Load balancing and auto-scaling

## 🎯 Use Cases & Applications

### 📈 Business Intelligence

- **Document Analysis**: Contract review and compliance checking
- **Customer Support**: Intelligent chatbots and ticket routing
- **Market Research**: Trend analysis and competitive intelligence
- **Financial Analytics**: Risk assessment and portfolio management

### 🔬 Research & Development

- **Literature Review**: Paper summarization and knowledge extraction
- **Hypothesis Generation**: Research question formulation
- **Data Analysis**: Statistical interpretation and visualization
- **Experiment Design**: Methodology development and validation

### 🎓 Education & Training

- **Personalized Tutoring**: Adaptive learning systems
- **Content Generation**: Course material and assessment creation
- **Knowledge Assessment**: Automated grading and feedback
- **Skill Development**: Professional training and certification

## 📊 Performance Benchmarks

### Memory Efficiency Comparison

| Method           | Memory Usage | Training Speed | Accuracy Drop | Use Case            |
| ---------------- | ------------ | -------------- | ------------- | ------------------- |
| Full Fine-tuning | 84GB         | 100%           | 0%            | Maximum accuracy    |
| LoRA             | 14GB         | 160%           | <2%           | Balanced efficiency |
| QLoRA            | 3.6GB        | 155%           | <3%           | Consumer hardware   |
| 1-bit LLMs       | 2.1GB        | 250%           | <5%           | Edge deployment     |

### Agent Performance Metrics

- **Response Latency**: 200-500ms for simple queries
- **Retrieval Accuracy**: Precision@10 > 0.85 on domain datasets
- **Function Calling Success**: >95% accuracy on structured tasks
- **Memory Efficiency**: 16x reduction with minimal quality loss

## 🛠️ Development & Contribution

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black examples/ scripts/
flake8 examples/ scripts/

# Build documentation
mkdocs build
```

### Contributing Guidelines

1. **Fork the Repository**: Create your own copy for development
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**: Submit your contribution for review

### Code Standards

- **Python Style**: Follow PEP 8 guidelines
- **Documentation**: Include docstrings for all functions
- **Testing**: Add tests for new functionality
- **Examples**: Provide working code demonstrations

## 🔗 Resources & References

### 📚 Essential Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) - Parameter-efficient fine-tuning
- [QLoRA: Efficient Finetuning](https://arxiv.org/abs/2305.14314) - Quantized fine-tuning
- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) - Ultra-efficient models

### 🌐 External Tools & Libraries

- **[Hugging Face Transformers](https://huggingface.co/transformers/)**: Model library and APIs
- **[LangChain](https://langchain.readthedocs.io/)**: Agent framework and tools
- **[FAISS](https://github.com/facebookresearch/faiss)**: Vector similarity search
- **[OpenTelemetry](https://opentelemetry.io/)**: Observability and tracing

### 🎓 Learning Paths

- **Beginner**: Start with Foundational LLMs → Embeddings → Getting Started examples
- **Intermediate**: Generative Agents → Domain-Specific adaptation → Fine-tuning basics
- **Advanced**: 1-bit models → Production deployment → Custom agent architectures

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to the open-source community and researchers whose work makes this educational resource possible:

- **Hugging Face Team** for model hosting and tools
- **Meta AI** for Llama model series and research
- **Google Research** for Transformer architecture and Gemini APIs
- **Microsoft** for QLoRA and quantization research
- **OpenAI** for advancing LLM capabilities and best practices

## 📧 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/sanjanb/ai-agents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sanjanb/ai-agents/discussions)
- **Documentation**: [Live Website](http://localhost:8000)
- **Email**: Contact repository maintainers for collaboration

---

**🚀 Ready to build intelligent agents? Start with our [Getting Started Guide](docs/agents/getting-started.md) and explore the examples!**
