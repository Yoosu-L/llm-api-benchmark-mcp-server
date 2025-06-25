import random
import string
import time

MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 10

def generate_random_word() -> str:
    """
    Generates a random word with a length between MIN_WORD_LENGTH and MAX_WORD_LENGTH.
    """
    word_length = random.randint(MIN_WORD_LENGTH, MAX_WORD_LENGTH)
    word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
    return word

def generate_random_phrase(num_words: int) -> str:
    """
    Generates a random phrase consisting of num_words random words.
    """
    random.seed(time.time_ns()) # Seed with nanoseconds for better randomness
    
    random_words = [generate_random_word() for _ in range(num_words)]
    random_phrase = ' '.join(random_words)
    
    result = "Please reply back the following section unchanged: " + random_phrase
    return result