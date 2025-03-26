from .alpaca import AlpacaEvaluator
from .llama_qst import LlamaQuestionsEvaluator
from .reasoning import ReasoningQAEvaluator
from .trivia_qa import TriviaQAEvaluator
from .web_qst import WebQuestionsEvaluator

evaluator_mapping = {
    'alpaca_eval': AlpacaEvaluator,
    'llama_questions': LlamaQuestionsEvaluator,
    'reasoning_qa': ReasoningQAEvaluator,
    'trivia_qa': TriviaQAEvaluator,
    'web_questions': WebQuestionsEvaluator,
}