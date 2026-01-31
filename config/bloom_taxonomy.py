# =============================================================================
# SMART EXAM – Bloom's Taxonomy Configuration
# Fichier : config/bloom_taxonomy.py
# À QUOI SERT CE FICHIER :
# Définit les 6 niveaux de Bloom + liste exhaustive de verbes d'action
# pour chaque niveau (minimum 10 verbes par niveau).
# Utilisé par le Validator et les Agents.
# =============================================================================

BLOOM_TAXONOMY = {
    "Remember": {
        "level": 1,
        "description": "Recall facts and basic concepts",
        "verbs": [
            "list", "define", "recall", "name", "identify",
            "recognize", "reproduce", "state", "memorize",
            "quote", "retrieve", "recount", "repeat", "cite"
        ],
        "question_types": [
            "Fill in the blank",
            "Multiple choice (recall)",
            "True/False",
            "List the...",
            "Define the term..."
        ]
    },
    "Understand": {
        "level": 2,
        "description": "Explain ideas or concepts",
        "verbs": [
            "explain", "summarize", "describe", "interpret",
            "give examples", "paraphrase", "classify", "discuss",
            "clarify", "translate", "restate", "convert", "predict",
            "infer", "compare"
        ],
        "question_types": [
            "Explain what...",
            "Summarize...",
            "Describe in your own words...",
            "Give an example of...",
            "Compare and contrast..."
        ]
    },
    "Apply": {
        "level": 3,
        "description": "Use information in a new situation",
        "verbs": [
            "solve", "use", "implement", "calculate", "demonstrate",
            "apply", "show", "execute", "construct", "modify",
            "discover", "produce", "prepare", "choose", "complete"
        ],
        "question_types": [
            "Solve this problem...",
            "Use the concept to...",
            "How would you use...",
            "Calculate...",
            "Design a solution..."
        ]
    },
    "Analyze": {
        "level": 4,
        "description": "Draw connections among ideas",
        "verbs": [
            "compare", "contrast", "differentiate", "classify",
            "organize", "analyze", "break down", "distinguish",
            "categorize", "order", "examine", "inspect", "test",
            "separate", "investigate"
        ],
        "question_types": [
            "Compare and contrast...",
            "Classify...",
            "What is the difference between...",
            "Analyze the structure of...",
            "Categorize..."
        ]
    },
    "Evaluate": {
        "level": 5,
        "description": "Justify a stand or decision",
        "verbs": [
            "justify", "critique", "assess", "defend", "recommend",
            "evaluate", "decide", "judge", "argue", "support",
            "select", "choose", "debate", "verify", "criticize"
        ],
        "question_types": [
            "Evaluate the effectiveness of...",
            "Justify your choice of...",
            "Critique the argument that...",
            "Do you agree with...? Why?",
            "What is the best solution and why?"
        ]
    },
    "Create": {
        "level": 6,
        "description": "Produce new or original work",
        "verbs": [
            "design", "invent", "propose", "create", "develop",
            "plan", "construct", "generate", "formulate", "compose",
            "devise", "write", "build", "synthesize", "combine"
        ],
        "question_types": [
            "Design a new...",
            "Invent a way to...",
            "What would you create if...",
            "Propose a solution for...",
            "Develop a plan to..."
        ]
    }
}

def get_bloom_level(level_name):
    """Get Bloom level details by name"""
    return BLOOM_TAXONOMY.get(level_name)

def get_all_levels():
    """Get all Bloom levels"""
    return BLOOM_TAXONOMY

def get_verbs_for_level(level_name):
    """Get action verbs for a specific Bloom level"""
    level = BLOOM_TAXONOMY.get(level_name)
    return level["verbs"] if level else []

def get_question_types_for_level(level_name):
    """Get question types for a specific Bloom level"""
    level = BLOOM_TAXONOMY.get(level_name)
    return level["question_types"] if level else []