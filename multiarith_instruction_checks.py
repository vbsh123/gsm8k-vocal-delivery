"""Observable instruction-following measures shared by the archived and new answers."""
import re

THIRD_ASK = 'Begin your response with the word Ready.'
START = re.compile(r'^\s*Ready\b', re.I)
NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
FINAL_LINE = re.compile(r'#### (?:' + NUMBER + r')(?:/\d+)?')
# A written operation, whether or not the calculation gives the right result.
OPERATION = re.compile(r'\d\s*(?:[+*/×÷−-]|\\times\b|\\cdot\b|\\div\b|plus\b|minus\b|times\b|divided by\b)'
                       r'\s*[(\s$]*[+-]?\d', re.I)


def checks(response):
    text = response.strip()
    last_line = text.splitlines()[-1] if text else ''
    opening = bool(START.match(text))
    final = bool(FINAL_LINE.fullmatch(last_line))
    reasoning_text = text.rsplit('####', 1)[0]
    operation = bool(OPERATION.search(reasoning_text))
    return {'ready-opening': opening, 'written-operation': operation,
            'exact-final-line': final, 'all-three-observed': opening and operation and final}
