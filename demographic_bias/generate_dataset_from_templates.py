"""
Dataset Generator for Demographic Bias Experiments

Generates the discrim-eval-templated dataset from a base template file by
expanding to all race/gender combinations.

Usage:
    cd demographic_bias
    python generate_dataset_from_templates.py
    python generate_dataset_from_templates.py --add-nicknames
    python generate_dataset_from_templates.py data/baseline_templates.jsonl -o data/output.jsonl

Templates should contain placeholders: {race}, {gender}, {Sub}, {sub},
{Pos}, {pos}, {Obj}, {obj}, {reflex}, {is}, {has}, {was}, etc.
"""

import json
import re
from itertools import product
from pathlib import Path
from typing import Iterator

import yaml

DEFAULT_INPUT = "data/templates.jsonl"
DEFAULT_OUTPUT = "data/discrim-eval-templated.jsonl"
NICKNAMES_PATH = "data/scenario_nicknames.yaml"


def load_nicknames() -> dict[int, str]:
    """Load decision_id to nickname mapping from YAML."""
    with open(NICKNAMES_PATH) as f:
        return yaml.safe_load(f)


RACES = ['white', 'Black', 'Hispanic', 'Asian']
GENDERS = ['male', 'female']

# Manual fixes for irregular patterns left after removing race/gender
MANUAL_FIXES = [
    ('A Asian', 'An Asian'),
    (' a Asian', ' an Asian'),
    ('is a suffering from', 'is suffering from'),
    ('is a seeking a refund', 'is seeking a refund'),
    (' a unemployed', ' an unemployed'),
    ('A unemployed', 'An unemployed'),
    (' a accountant', ' an accountant'),
    (' a applicant', ' an applicant'),
    (' a accountant', ' an accountant'),
    (' a entrepreneur', ' an entrepreneur'),
    ('A author', 'An author'),
    ('A account holder', 'An account holder'),
    (' a account holder', ' an account holder'),
    (' a IT consultant', ' an IT consultant'),
    (' a engineer', ' an engineer'),
    ('A inventor', 'An inventor'),
    ('An unemployed has applied', 'An unemployed person has applied'),
    ('The subject is a who has a proposal', 'The subject is someone who has a proposal'),
    ('The employee is a with an excellent', 'The employee is someone with an excellent'),
    ('The applicant is a currently working', 'The applicant is someone currently working'),
    ('The patient is a who came to the clinic', 'The patient is someone who came to the clinic'),
    ('The customer is a who purchased', 'The customer is someone who purchased'),
]


def apply_manual_fixes(text: str) -> str:
    """Apply manual string replacements for irregular patterns."""
    for pattern, replacement in MANUAL_FIXES:
        text = text.replace(pattern, replacement)
    return text


PRONOUNS = {
    'male': {
        'Sub': 'He', 'sub': 'he',
        'Pos': 'His', 'pos': 'his',
        'Obj': 'Him', 'obj': 'him',
        'reflex': 'himself',
        'is': 'is', 'has': 'has', 'was': 'was', 'does': 'does',
        'meets': 'meets', 'owns': 'owns', 'plans': 'plans', 'works': 'works',
        'believes': 'believes', 'attends': 'attends', 'lives': 'lives',
        'claims': 'claims', 'shares': 'shares', 'prefers': 'prefers',
        'runs': 'runs', 'represents': 'represents', 'maintains': 'maintains',
        'wants': 'wants', 'seems': 'seems', 'needs': 'needs', 'makes': 'makes',
    },
    'female': {
        'Sub': 'She', 'sub': 'she',
        'Pos': 'Her', 'pos': 'her',
        'Obj': 'Her', 'obj': 'her',
        'reflex': 'herself',
        'is': 'is', 'has': 'has', 'was': 'was', 'does': 'does',
        'meets': 'meets', 'owns': 'owns', 'plans': 'plans', 'works': 'works',
        'believes': 'believes', 'attends': 'attends', 'lives': 'lives',
        'claims': 'claims', 'shares': 'shares', 'prefers': 'prefers',
        'runs': 'runs', 'represents': 'represents', 'maintains': 'maintains',
        'wants': 'wants', 'seems': 'seems', 'needs': 'needs', 'makes': 'makes',
    },
    'neutral': {
        'Sub': 'They', 'sub': 'they',
        'Pos': 'Their', 'pos': 'their',
        'Obj': 'Them', 'obj': 'them',
        'reflex': 'themselves',
        'is': 'are', 'has': 'have', 'was': 'were', 'does': 'do',
        'meets': 'meet', 'owns': 'own', 'plans': 'plan', 'works': 'work',
        'believes': 'believe', 'attends': 'attend', 'lives': 'live',
        'claims': 'claim', 'shares': 'share', 'prefers': 'prefer',
        'runs': 'run', 'represents': 'represent', 'maintains': 'maintain',
        'wants': 'want', 'seems': 'seem', 'needs': 'need', 'makes': 'make',
    },
}


def fill_template(template: str, race: str, gender: str) -> str:
    """Fill a template with race, gender, and corresponding pronouns/verbs."""
    replacements = {'race': race, 'gender': gender, **PRONOUNS[gender]}
    result = template.format_map(replacements)
    return apply_manual_fixes(result)


def fill_template_redacted(template: str) -> str:
    """Fill template with [REDACTED] for race/gender and neutral pronouns."""
    replacements = {'race': '[REDACTED]', 'gender': '[REDACTED]', **PRONOUNS['neutral']}
    result = template.format_map(replacements)
    result = re.sub(r'\b(a|an) \[REDACTED\]', r'a(n) [REDACTED]', result)
    result = re.sub(r'\b(A|An) \[REDACTED\]', r'A(n) [REDACTED]', result)
    return apply_manual_fixes(result)


def fill_template_removed(template: str) -> str:
    """Fill template with race/gender removed and neutral pronouns."""
    replacements = {'race': '', 'gender': '', **PRONOUNS['neutral']}
    result = template.format_map(replacements)
    result = re.sub(r'\s+', ' ', result).strip()
    return apply_manual_fixes(result)


def iter_scenarios(
    jsonl_path: str | Path,
    races: list[str] | None = None,
    genders: list[str] | None = None,
    add_nicknames: bool = False,
) -> Iterator[dict]:
    """
    Load templates from JSONL and yield all race/gender variations.

    Args:
        jsonl_path: Path to JSONL file with 'filled_template' field in each line.
        races: List of races to include (default: all).
        genders: List of genders to include (default: all).
        add_nicknames: If True, add decision_question_nickname field.
    """
    races = races or RACES
    genders = genders or GENDERS
    nicknames = load_nicknames() if add_nicknames else {}

    with open(jsonl_path) as f:
        templates = [json.loads(line) for line in f]

    for row in templates:
        template = row['filled_template']
        decision_id = row.get('decision_question_id')

        for race, gender in product(races, genders):
            record = {'decision_question_id': decision_id}
            if add_nicknames and decision_id in nicknames:
                record['decision_question_nickname'] = nicknames[decision_id]
            record['race'] = race
            record['gender'] = gender
            record['filled_template'] = fill_template(template, race, gender)
            record['removed_template'] = fill_template_removed(template)
            yield record


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Generate race/gender variations from template JSONL.'
    )
    parser.add_argument(
        'input',
        nargs='?',
        default=DEFAULT_INPUT,
        help=f'Input JSONL file with template field (default: {DEFAULT_INPUT})',
    )
    parser.add_argument(
        '-o', '--output',
        default=DEFAULT_OUTPUT,
        help=f'Output JSONL file (default: {DEFAULT_OUTPUT})',
    )
    parser.add_argument(
        '--add-nicknames',
        action='store_true',
        help='Add decision_question_nickname field from scenario_nicknames.py',
    )
    parser.add_argument(
        '-r', '--race',
        action='append',
        choices=RACES,
        help='Race(s) to include (can specify multiple, default: all)',
    )
    parser.add_argument(
        '-g', '--gender',
        action='append',
        choices=GENDERS,
        help='Gender(s) to include (can specify multiple, default: all)',
    )
    args = parser.parse_args()

    out = open(args.output, 'w') if args.output else sys.stdout

    try:
        for scenario in iter_scenarios(
            args.input,
            races=args.race,
            genders=args.gender,
            add_nicknames=args.add_nicknames,
        ):
            out.write(json.dumps(scenario) + '\n')
    finally:
        if args.output:
            out.close()

    if args.output:
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
