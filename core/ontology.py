"""
EXAI-ResumeIntel: Skill ontology and concept expansion engine
==============================================================

Maps surface-level terms to canonical skill concepts and handles implicit
skill inference (e.g. YOLO -> Computer Vision -> Machine Learning). The
ontology contains 22 canonical skill nodes and 346 alias strings across
IT, Finance, HR, Healthcare, Legal, and other domains.

Corresponds to paper Section III.C (Domain Ontology and Skill Extraction).

Module: core.ontology
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""


import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SKILL ONTOLOGY
# Structure: canonical_skill → { aliases, children, parents, weight }
# ─────────────────────────────────────────────────────────────────────────────

SKILL_ONTOLOGY: Dict[str, Dict] = {

    # ── INFORMATION TECHNOLOGY ──────────────────────────────────────────────
    "machine_learning": {
        "aliases": [
            "ml", "machine learning", "deep learning", "dl", "artificial intelligence",
            "ai", "neural network", "neural networks", "ann", "dnn", "mlops",
            "supervised learning", "unsupervised learning", "reinforcement learning",
            "rl", "transfer learning", "few shot learning", "zero shot", "self supervised"
        ],
        "children": [
            "computer_vision", "nlp", "time_series", "recommendation_systems",
            "model_deployment", "feature_engineering"
        ],
        "parents": [],
        "weight": 1.0,
        "category": "INFORMATION-TECHNOLOGY"
    },
    "computer_vision": {
        "aliases": [
            "computer vision", "cv", "image processing", "image recognition",
            "object detection", "object recognition", "image segmentation",
            "yolo", "yolov5", "yolov7", "yolov8", "yolo v5", "yolo v8",
            "coco", "coco dataset", "imagenet", "voc dataset",
            "resnet", "vgg", "efficientnet", "mobilenet", "inception",
            "faster rcnn", "mask rcnn", "ssd", "retinanet", "detr",
            "semantic segmentation", "instance segmentation", "panoptic",
            "anchor boxes", "bounding box", "iou", "map", "mean average precision",
            "feature pyramid", "fpn", "roi", "nms", "non max suppression",
            "optical flow", "pose estimation", "depth estimation", "stereo vision",
            "mediapipe", "opencv", "cv2", "pillow", "pil", "albumentations",
            "augmentation", "data augmentation", "torchvision", "detectron2",
            "mmdetection", "ultralytics", "roboflow", "labelimg", "cvat",
            "gan", "generative adversarial", "stable diffusion", "dalle",
            "clip", "sam", "segment anything", "grounding dino"
        ],
        "children": [],
        "parents": ["machine_learning"],
        "weight": 0.95,
        "category": "INFORMATION-TECHNOLOGY"
    },
    "nlp": {
        "aliases": [
            "nlp", "natural language processing", "natural language understanding",
            "nlu", "nlg", "natural language generation", "text mining",
            "text classification", "sentiment analysis", "named entity recognition",
            "ner", "pos tagging", "dependency parsing", "coreference resolution",
            "question answering", "qa", "information extraction",
            "bert", "roberta", "gpt", "gpt2", "gpt3", "gpt4", "chatgpt",
            "llm", "large language model", "transformers", "attention mechanism",
            "self attention", "huggingface", "hugging face", "langchain",
            "llamaindex", "llama", "mistral", "claude", "gemini",
            "tokenization", "word embeddings", "word2vec", "glove", "fasttext",
            "sentence transformers", "sbert", "semantic similarity",
            "text summarization", "machine translation", "speech recognition",
            "asr", "tts", "text to speech", "spacy", "nltk", "gensim",
            "regex", "regular expression", "stemming", "lemmatization",
            "bag of words", "bow", "tfidf", "tf idf", "n gram",
            "rag", "retrieval augmented generation", "vector database",
            "pinecone", "weaviate", "chroma", "faiss", "embedding"
        ],
        "children": [],
        "parents": ["machine_learning"],
        "weight": 0.95,
        "category": "INFORMATION-TECHNOLOGY"
    },
    "deep_learning_frameworks": {
        "aliases": [
            "pytorch", "torch", "tensorflow", "tf", "keras", "jax", "flax",
            "mxnet", "caffe", "theano", "paddle", "paddlepaddle",
            "onnx", "tensorrt", "openvino", "coreml", "tflite",
            "cuda", "cudnn", "gpu programming", "gpu acceleration",
            "distributed training", "data parallel", "model parallel",
            "apex", "deepspeed", "accelerate", "lightning", "pytorch lightning",
            "gradient descent", "backpropagation", "batch normalization",
            "dropout", "regularization", "hyperparameter tuning", "optuna",
            "ray tune", "wandb", "weights and biases", "mlflow", "tensorboard"
        ],
        "children": [],
        "parents": ["machine_learning"],
        "weight": 0.9,
        "category": "INFORMATION-TECHNOLOGY"
    },
    "data_science": {
        "aliases": [
            "data science", "data scientist", "data analysis", "data analytics",
            "exploratory data analysis", "eda", "statistical analysis", "statistics",
            "hypothesis testing", "a/b testing", "ab testing", "regression",
            "classification", "clustering", "k means", "dbscan", "pca",
            "dimensionality reduction", "feature selection", "model evaluation",
            "cross validation", "overfitting", "underfitting", "bias variance",
            "confusion matrix", "roc curve", "auc", "precision recall",
            "f1 score", "accuracy", "mse", "rmse", "mae", "r squared",
            "pandas", "numpy", "scipy", "statsmodels", "pingouin",
            "matplotlib", "seaborn", "plotly", "bokeh", "altair", "ggplot",
            "jupyter", "jupyter notebook", "google colab", "kaggle",
            "scikit learn", "sklearn", "xgboost", "lightgbm", "catboost",
            "random forest", "gradient boosting", "decision tree", "svm",
            "support vector machine", "naive bayes", "knn", "logistic regression",
            "linear regression", "ensemble methods", "bagging", "boosting",
            "time series", "arima", "prophet", "forecasting", "anomaly detection"
        ],
        "children": ["machine_learning"],
        "parents": [],
        "weight": 1.0,
        "category": "INFORMATION-TECHNOLOGY"
    },
    "python": {
        "aliases": [
            "python", "python3", "python 3", "py", "cpython", "ipython",
            "pip", "conda", "virtualenv", "venv", "poetry", "pipenv",
            "pytest", "unittest", "mock", "fixtures", "tox",
            "asyncio", "async await", "multiprocessing", "threading",
            "decorators", "generators", "comprehensions", "type hints",
            "dataclasses", "pydantic", "mypy", "black", "flake8", "isort"
        ],
        "children": [],
        "parents": ["data_science", "software_engineering"],
        "weight": 0.9,
        "category": "INFORMATION-TECHNOLOGY"
    },
    "software_engineering": {
        "aliases": [
            "software engineering", "software development", "programming",
            "software developer", "full stack", "fullstack", "backend", "frontend",
            "web development", "application development", "system design",
            "object oriented", "oop", "functional programming", "design patterns",
            "solid principles", "clean code", "refactoring", "code review",
            "agile", "scrum", "kanban", "sprint", "jira", "confluence",
            "git", "github", "gitlab", "bitbucket", "version control",
            "ci/cd", "continuous integration", "continuous deployment",
            "devops", "docker", "kubernetes", "k8s", "helm", "terraform",
            "aws", "azure", "gcp", "cloud", "microservices", "rest api",
            "graphql", "grpc", "soap", "api design", "swagger", "openapi"
        ],
        "children": ["python", "web_development", "databases"],
        "parents": [],
        "weight": 1.0,
        "category": "INFORMATION-TECHNOLOGY"
    },
    "web_development": {
        "aliases": [
            "html", "css", "javascript", "js", "typescript", "ts",
            "react", "reactjs", "react.js", "vue", "vuejs", "angular",
            "nextjs", "next.js", "nuxt", "svelte", "gatsby",
            "nodejs", "node.js", "express", "fastapi", "flask", "django",
            "spring", "spring boot", "rails", "laravel", "asp.net",
            "tailwind", "bootstrap", "sass", "scss", "webpack", "vite",
            "rest", "restful", "http", "https", "websocket", "graphql",
            "redux", "mobx", "zustand", "context api", "hooks",
            "responsive design", "mobile first", "pwa", "spa"
        ],
        "children": [],
        "parents": ["software_engineering"],
        "weight": 0.9,
        "category": "INFORMATION-TECHNOLOGY"
    },
    "databases": {
        "aliases": [
            "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
            "sql server", "mssql", "nosql", "mongodb", "cassandra", "redis",
            "elasticsearch", "dynamodb", "firebase", "supabase",
            "database design", "schema design", "normalization", "indexing",
            "query optimization", "stored procedures", "triggers", "views",
            "data warehouse", "olap", "etl", "data pipeline", "airflow",
            "spark", "hadoop", "hive", "kafka", "flink", "dbt",
            "snowflake", "bigquery", "redshift", "databricks", "delta lake"
        ],
        "children": [],
        "parents": ["software_engineering", "data_science"],
        "weight": 0.85,
        "category": "INFORMATION-TECHNOLOGY"
    },
    "cybersecurity": {
        "aliases": [
            "cybersecurity", "cyber security", "information security", "infosec",
            "network security", "penetration testing", "pentest", "ethical hacking",
            "vulnerability assessment", "risk assessment", "soc", "siem",
            "firewall", "ids", "ips", "intrusion detection", "threat analysis",
            "incident response", "forensics", "malware analysis", "reverse engineering",
            "encryption", "cryptography", "pki", "ssl", "tls", "vpn",
            "owasp", "cve", "cvss", "zero day", "exploit", "payload",
            "metasploit", "burp suite", "nmap", "wireshark", "kali linux",
            "iso 27001", "nist", "gdpr", "compliance", "audit"
        ],
        "children": [],
        "parents": [],
        "weight": 1.0,
        "category": "INFORMATION-TECHNOLOGY"
    },
    "model_deployment": {
        "aliases": [
            "model deployment", "mlops", "model serving", "model monitoring",
            "model registry", "feature store", "a/b testing ml", "shadow mode",
            "canary deployment", "blue green", "fastapi", "flask api",
            "torchserve", "triton", "bentoml", "seldon", "kubeflow",
            "sagemaker", "vertex ai", "azure ml", "databricks mlflow",
            "batch inference", "real time inference", "streaming inference",
            "model compression", "quantization", "pruning", "distillation",
            "onnx export", "tensorrt optimization", "edge deployment"
        ],
        "children": [],
        "parents": ["machine_learning", "software_engineering"],
        "weight": 0.9,
        "category": "INFORMATION-TECHNOLOGY"
    },

    # ── FINANCE & ACCOUNTING ─────────────────────────────────────────────────
    "accounting": {
        "aliases": [
            "accounting", "accountant", "bookkeeping", "general ledger", "gl",
            "accounts payable", "ap", "accounts receivable", "ar",
            "financial statements", "balance sheet", "income statement", "p&l",
            "cash flow", "journal entries", "reconciliation", "trial balance",
            "gaap", "ifrs", "accrual", "depreciation", "amortization",
            "tax", "taxation", "gst", "vat", "tds", "income tax",
            "audit", "internal audit", "statutory audit", "sox",
            "tally", "quickbooks", "sap fico", "oracle financials",
            "cost accounting", "management accounting", "budget", "variance analysis"
        ],
        "children": [],
        "parents": ["finance"],
        "weight": 0.95,
        "category": "ACCOUNTANT"
    },
    "finance": {
        "aliases": [
            "finance", "financial analysis", "financial modeling", "dcf",
            "valuation", "equity research", "investment banking", "ib",
            "private equity", "pe", "venture capital", "vc",
            "portfolio management", "asset management", "wealth management",
            "risk management", "derivatives", "options", "futures", "hedging",
            "forex", "fixed income", "bonds", "equities", "mutual funds",
            "cfa", "frm", "cpa", "ca", "acca", "cma",
            "bloomberg", "reuters", "excel financial modeling", "vba finance",
            "financial reporting", "investor relations", "m&a", "due diligence"
        ],
        "children": ["accounting"],
        "parents": [],
        "weight": 1.0,
        "category": "FINANCE"
    },
    "banking": {
        "aliases": [
            "banking", "retail banking", "commercial banking", "investment banking",
            "credit analysis", "loan", "underwriting", "credit risk",
            "kyc", "aml", "anti money laundering", "compliance banking",
            "trade finance", "treasury", "liquidity", "basel", "rbi",
            "sebi", "fd", "loan processing", "npa", "credit scoring",
            "fintech", "payment systems", "upi", "swift", "correspondent banking"
        ],
        "children": [],
        "parents": ["finance"],
        "weight": 0.95,
        "category": "BANKING"
    },

    # ── HEALTHCARE ───────────────────────────────────────────────────────────
    "healthcare": {
        "aliases": [
            "healthcare", "medical", "clinical", "patient care", "nursing",
            "physician", "doctor", "surgeon", "diagnosis", "treatment",
            "ehr", "emr", "epic", "cerner", "meditech", "health informatics",
            "hipaa", "icd", "cpt codes", "medical coding", "billing medical",
            "pharmacology", "anatomy", "physiology", "pathology",
            "radiology", "mri", "ct scan", "ultrasound", "x ray",
            "public health", "epidemiology", "clinical trials", "research",
            "telemedicine", "telehealth", "patient engagement", "care coordination"
        ],
        "children": [],
        "parents": [],
        "weight": 1.0,
        "category": "HEALTHCARE"
    },

    # ── LEGAL ────────────────────────────────────────────────────────────────
    "legal": {
        "aliases": [
            "legal", "law", "attorney", "lawyer", "advocate", "counsel",
            "litigation", "contract", "contract drafting", "legal research",
            "case law", "precedent", "court", "tribunal", "arbitration", "mediation",
            "corporate law", "mergers acquisitions", "ipr", "intellectual property",
            "patent", "trademark", "copyright", "criminal law", "civil law",
            "family law", "labor law", "employment law", "tax law",
            "compliance legal", "regulatory", "due diligence legal", "legal drafting",
            "pleadings", "discovery", "deposition", "brief writing"
        ],
        "children": [],
        "parents": [],
        "weight": 1.0,
        "category": "ADVOCATE"
    },

    # ── HUMAN RESOURCES ──────────────────────────────────────────────────────
    "hr": {
        "aliases": [
            "human resources", "hr", "talent acquisition", "recruitment",
            "talent management", "onboarding", "offboarding", "employee relations",
            "performance management", "compensation", "benefits", "payroll",
            "hris", "workday", "successfactors", "bamboohr", "adp",
            "learning development", "l&d", "training", "organizational development",
            "diversity inclusion", "dei", "culture", "engagement",
            "labor law hr", "fmla", "eeo", "workforce planning", "succession planning"
        ],
        "children": [],
        "parents": [],
        "weight": 1.0,
        "category": "HR"
    },

    # ── SALES & MARKETING ────────────────────────────────────────────────────
    "sales": {
        "aliases": [
            "sales", "business development", "b2b sales", "b2c sales",
            "account management", "key account", "crm", "salesforce", "hubspot",
            "pipeline management", "lead generation", "prospecting",
            "cold calling", "negotiation", "closing", "upselling", "cross selling",
            "revenue", "quota", "territory management", "channel sales",
            "inside sales", "field sales", "enterprise sales", "saas sales"
        ],
        "children": [],
        "parents": [],
        "weight": 1.0,
        "category": "SALES"
    },
    "marketing": {
        "aliases": [
            "marketing", "digital marketing", "content marketing", "seo", "sem",
            "social media marketing", "smm", "email marketing", "ppc",
            "google ads", "facebook ads", "meta ads", "instagram", "linkedin",
            "brand management", "brand strategy", "market research",
            "consumer insights", "product marketing", "growth hacking",
            "conversion optimization", "cro", "funnel", "customer acquisition",
            "retention marketing", "lifecycle marketing", "analytics marketing",
            "google analytics", "ga4", "mixpanel", "amplitude", "hotjar"
        ],
        "children": [],
        "parents": [],
        "weight": 1.0,
        "category": "DIGITAL-MEDIA"
    },

    # ── ENGINEERING ──────────────────────────────────────────────────────────
    "mechanical_engineering": {
        "aliases": [
            "mechanical engineering", "cad", "solidworks", "autocad", "catia",
            "ansys", "fea", "finite element analysis", "cfd",
            "manufacturing", "machining", "cnc", "lean manufacturing", "six sigma",
            "product design", "prototyping", "3d printing", "additive manufacturing",
            "thermodynamics", "fluid mechanics", "material science",
            "quality control", "quality assurance", "iso 9001", "tolerances"
        ],
        "children": [],
        "parents": [],
        "weight": 1.0,
        "category": "ENGINEERING"
    },
    "civil_engineering": {
        "aliases": [
            "civil engineering", "structural engineering", "structural analysis",
            "staad pro", "etabs", "autocad civil", "revit", "bim",
            "construction management", "project planning", "scheduling",
            "concrete", "steel", "foundation", "geotechnical", "surveying",
            "quantity estimation", "boc", "mos", "roads", "bridges",
            "water treatment", "wastewater", "environmental engineering"
        ],
        "children": [],
        "parents": [],
        "weight": 1.0,
        "category": "CONSTRUCTION"
    },

    # ── DESIGN ───────────────────────────────────────────────────────────────
    "design": {
        "aliases": [
            "graphic design", "ui design", "ux design", "ui/ux", "product design",
            "visual design", "interaction design", "figma", "sketch", "adobe xd",
            "photoshop", "illustrator", "indesign", "premiere pro", "after effects",
            "prototyping", "wireframing", "user research", "usability testing",
            "design thinking", "design system", "brand identity", "typography",
            "color theory", "motion design", "3d design", "blender", "cinema 4d",
            "invision", "zeplin", "framer", "principle", "maze", "usertesting"
        ],
        "children": [],
        "parents": [],
        "weight": 1.0,
        "category": "DESIGNER"
    },

    # ── SOFT SKILLS ──────────────────────────────────────────────────────────
    "leadership": {
        "aliases": [
            "leadership", "team management", "people management", "mentoring",
            "coaching", "cross functional", "stakeholder management",
            "executive presence", "strategic thinking", "decision making",
            "change management", "conflict resolution", "negotiation soft"
        ],
        "children": [],
        "parents": [],
        "weight": 0.7,
        "category": "ALL"
    },
    "communication": {
        "aliases": [
            "communication", "presentation", "public speaking", "written communication",
            "verbal communication", "storytelling", "documentation", "technical writing",
            "report writing", "business writing", "client communication"
        ],
        "children": [],
        "parents": [],
        "weight": 0.6,
        "category": "ALL"
    },
    "project_management": {
        "aliases": [
            "project management", "pmp", "prince2", "agile pm", "waterfall",
            "risk management pm", "resource planning", "milestone", "gantt",
            "ms project", "asana", "trello", "monday.com", "basecamp",
            "budget management", "deliverables", "stakeholders", "timeline"
        ],
        "children": [],
        "parents": [],
        "weight": 0.75,
        "category": "ALL"
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# JOB ROLE → REQUIRED SKILL WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

ROLE_SKILL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "INFORMATION-TECHNOLOGY": {
        "machine_learning": 0.8,
        "software_engineering": 0.9,
        "databases": 0.8,
        "python": 0.85,
        "web_development": 0.7,
        "cybersecurity": 0.5,
        "communication": 0.5,
        "project_management": 0.5,
    },
    "MACHINE LEARNING ENGINEER": {
        "machine_learning": 1.0,
        "deep_learning_frameworks": 1.0,
        "computer_vision": 0.8,
        "nlp": 0.8,
        "data_science": 0.9,
        "python": 1.0,
        "model_deployment": 0.9,
        "databases": 0.6,
        "software_engineering": 0.7,
        "communication": 0.4,
    },
    "DATA SCIENTIST": {
        "data_science": 1.0,
        "machine_learning": 0.9,
        "python": 1.0,
        "databases": 0.7,
        "deep_learning_frameworks": 0.7,
        "communication": 0.6,
        "project_management": 0.5,
    },
    "DEVOPS ENGINEER": {
        "software_engineering": 1.0,
        "databases": 0.5,
        "cybersecurity": 0.5,
        "project_management": 0.5,
        "communication": 0.5,
    },
    "DATA ENGINEER": {
        "databases": 1.0,
        "data_science": 0.7,
        "software_engineering": 0.7,
        "python": 0.8,
        "communication": 0.4,
    },
    "PRODUCT MANAGER": {
        "project_management": 0.9,
        "leadership": 0.8,
        "communication": 0.9,
        "data_science": 0.4,
        "software_engineering": 0.3,
    },
    "ACCOUNTANT": {
        "accounting": 1.0,
        "finance": 0.8,
        "communication": 0.6,
        "project_management": 0.4,
    },
    "FINANCE": {
        "finance": 1.0,
        "accounting": 0.7,
        "banking": 0.6,
        "data_science": 0.5,
        "communication": 0.7,
    },
    "BANKING": {
        "banking": 1.0,
        "finance": 0.9,
        "accounting": 0.6,
        "communication": 0.7,
        "leadership": 0.5,
    },
    "HEALTHCARE": {
        "healthcare": 1.0,
        "communication": 0.8,
        "leadership": 0.5,
        "project_management": 0.4,
    },
    "ADVOCATE": {
        "legal": 1.0,
        "communication": 0.9,
        "leadership": 0.6,
        "project_management": 0.5,
    },
    "HR": {
        "hr": 1.0,
        "communication": 0.9,
        "leadership": 0.8,
        "project_management": 0.6,
        "data_science": 0.3,
    },
    "SALES": {
        "sales": 1.0,
        "marketing": 0.7,
        "communication": 0.9,
        "leadership": 0.6,
    },
    "DIGITAL-MEDIA": {
        "marketing": 1.0,
        "design": 0.7,
        "communication": 0.9,
        "sales": 0.5,
    },
    "DESIGNER": {
        "design": 1.0,
        "communication": 0.7,
        "software_engineering": 0.4,
        "marketing": 0.5,
    },
    "ENGINEERING": {
        "mechanical_engineering": 0.9,
        "civil_engineering": 0.7,
        "software_engineering": 0.5,
        "project_management": 0.7,
        "communication": 0.5,
    },
    "CONSTRUCTION": {
        "civil_engineering": 1.0,
        "mechanical_engineering": 0.5,
        "project_management": 0.8,
        "communication": 0.6,
    },
    "TEACHER": {
        "communication": 1.0,
        "leadership": 0.8,
        "project_management": 0.6,
    },
    "CONSULTANT": {
        "communication": 0.9,
        "leadership": 0.8,
        "project_management": 0.8,
        "finance": 0.5,
        "data_science": 0.5,
    },
    "BUSINESS-DEVELOPMENT": {
        "sales": 0.9,
        "marketing": 0.7,
        "communication": 0.9,
        "leadership": 0.7,
        "finance": 0.4,
    },
    "PUBLIC-RELATIONS": {
        "communication": 1.0,
        "marketing": 0.8,
        "leadership": 0.6,
    },
    "ARTS": {
        "design": 0.8,
        "communication": 0.7,
    },
    "CHEF": {
        "communication": 0.7,
        "leadership": 0.6,
        "project_management": 0.5,
    },
    "FITNESS": {
        "communication": 0.8,
        "healthcare": 0.4,
        "leadership": 0.6,
    },
    "AVIATION": {
        "mechanical_engineering": 0.5,
        "communication": 0.8,
        "leadership": 0.7,
        "project_management": 0.6,
    },
    "AGRICULTURE": {
        "project_management": 0.6,
        "communication": 0.6,
    },
    "AUTOMOBILE": {
        "mechanical_engineering": 0.9,
        "software_engineering": 0.4,
        "project_management": 0.6,
    },
    "APPAREL": {
        "design": 0.7,
        "marketing": 0.6,
        "communication": 0.7,
    },
    "BPO": {
        "communication": 1.0,
        "leadership": 0.5,
        "software_engineering": 0.3,
    },
}


class OntologyEngine:
    """Fast lookup engine built from the ontology at init time."""

    def __init__(self):
        # alias → canonical_skill
        self.alias_map: Dict[str, str] = {}
        # canonical → all aliases + children aliases (expanded)
        self.skill_tokens: Dict[str, Set[str]] = defaultdict(set)

        self._build_index()

    def _build_index(self):
        for skill, data in SKILL_ONTOLOGY.items():
            for alias in data["aliases"]:
                norm = self._norm(alias)
                self.alias_map[norm] = skill
                self.skill_tokens[skill].add(norm)

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()

    def extract_skills(self, text: str) -> Dict[str, float]:
        """
        Extract canonical skills from raw text.
        Returns {skill: confidence_score}
        Handles multi-word phrases (up to 4-gram) for best coverage.
        """
        normed = self._norm(text)
        tokens = normed.split()
        found: Dict[str, float] = defaultdict(float)

        # n-gram sweep: 4→1 (longer matches get higher confidence)
        for n in range(4, 0, -1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                if phrase in self.alias_map:
                    skill = self.alias_map[phrase]
                    # longer match = more confident
                    conf = min(1.0, 0.6 + n * 0.1)
                    found[skill] = max(found[skill], conf)

        # propagate to parents (partial credit)
        expanded = dict(found)
        for skill, conf in found.items():
            for parent in SKILL_ONTOLOGY.get(skill, {}).get("parents", []):
                expanded[parent] = max(expanded.get(parent, 0.0), conf * 0.8)

        return expanded

    def get_role_requirements(self, role: str) -> Dict[str, float]:
        """Get required skills and weights for a job role."""
        role_upper = role.upper().replace("-", " ")
        # try exact match first
        for key in ROLE_SKILL_WEIGHTS:
            if key == role_upper or key.replace(" ", "") == role_upper.replace(" ", ""):
                return ROLE_SKILL_WEIGHTS[key]
        # fuzzy fallback
        for key in ROLE_SKILL_WEIGHTS:
            if any(word in key for word in role_upper.split() if len(word) > 3):
                return ROLE_SKILL_WEIGHTS[key]
        # default
        return {
            "communication": 0.8,
            "leadership": 0.6,
            "project_management": 0.5,
        }

    def get_skill_display_name(self, skill: str) -> str:
        return skill.replace("_", " ").title()

    def get_all_roles(self) -> List[str]:
        return sorted(ROLE_SKILL_WEIGHTS.keys())


# Singleton
_engine = None

def get_engine() -> OntologyEngine:
    global _engine
    if _engine is None:
        _engine = OntologyEngine()
    return _engine
