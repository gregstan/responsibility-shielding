"""
Robot participant experiment runner.

Each "participant" is one API conversation that replicates the Qualtrics survey flow:
welcome → cognitive load assignment → three vignettes (randomized order) → two proximate-agent
vignettes → cognitive load recall → comprehension checks → 2AFC → CRT → INDCOL.

Responses are written to a CSV whose column names match the human raw data export exactly,
so that the existing preprocessing.py pipeline can process robot data with zero modification.

After data collection, if run_analysis_after_collection=True, the script runs the same
core analysis functions used for human data (load_or_build_cleaned_dataframe,
compute_group_summaries, run_confirmatory_and_exploratory_tests, Tables 2/3/4/9) and
prints results to the terminal. All robot analysis outputs go to robot_raw_data/processed/
and robot_raw_data/tables/ — the human processed/ folder is never touched.

Usage:
    python robot_experiment/run_robot_participants.py

Configure the experiment by editing ROBOT_EXPERIMENT_CONFIG at the top of this file.
Set beta_mode=True for a small dry run that prints full transcripts.
Set overwrite_raw_data=False (the default) to append new rows to any existing raw data.
"""

import sys
from pathlib import Path

"Ensure project root is on sys.path so this script can be run directly"
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import csv
import json

"Load .env file if present — allows storing API keys in .env instead of system environment variables"
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass
import os
import random
import re
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Optional

from config import GeneralSettings, RobotSettings
from robot_experiment.model_clients import ModelClient, PROVIDER_MAX_TEMPERATURE, MODEL_TO_PROVIDER, get_client_for_model
from robot_experiment.stimuli import (
    ALL_RAW_RATING_COLUMNS,
    COGNITIVE_LOAD_HIGH_DIGIT,
    COGNITIVE_LOAD_INSTRUCTION_TEMPLATE,
    COGNITIVE_LOAD_LOW_DIGIT,
    COGNITIVE_LOAD_RECALL_PROMPT,
    COMPREHENSION_COLUMN_PREFIX,
    COMPREHENSION_QUESTIONS,
    CRT_QUESTIONS,
    DISTAL_BLAME_QUESTION,
    DISTAL_COLUMN_PREFIX,
    DISTAL_PUNISH_QUESTION,
    DISTAL_WRONG_QUESTION,
    INDCOL_ITEMS,
    PARTICIPANT_INSTRUCTIONS,
    PROXIMATE_BLAME_QUESTION,
    PROXIMATE_COLUMN_PREFIX,
    PROXIMATE_PREAMBLE,
    PROXIMATE_PUNISH_QUESTION,
    PROXIMATE_WRONG_QUESTION,
    TWO_AFC_COLUMN_PREFIX,
    TWO_AFC_QUESTIONS,
    VIGNETTE_BLOCK_INSTRUCTIONS,
    VIGNETTE_TEXT,
    distal_question_key,
)
from preprocessing import load_or_build_cleaned_dataframe
from core import compute_group_summaries, run_confirmatory_and_exploratory_tests
from tables import (
    compute_manuscript_table_2_mean_scale_values_by_dv_and_condition,
    compute_manuscript_table_3_primary_distal_blame_contrasts,
    compute_manuscript_table_4_story_specific_distal_blame_contrasts,
    compute_supplementary_table_9_secondary_dv_contrasts,
)


ROOT = Path(__file__).parent.parent


"============================================================"
"ROBOT EXPERIMENT CONFIGURATION — edit these before each run"
"============================================================"
ROBOT_EXPERIMENT_CONFIG: RobotSettings = {
    "models": ["claude-sonnet-4-6", "gpt-4o", "gemini-2.0-flash", "grok-3"],
    "n_participants_per_model": 1,
    "temperature": 1.0,
    "story_balance": "random",
    "output_file": str(ROOT / "robot_raw_data" / "robot_responsibility_shielding_raw.csv"),
    "max_concurrent_participants": 5,
    "beta_mode": False,
    "beta_n_participants": 3,
    "print_transcripts": True,
    "run_analysis_after_collection": True,
    "run_models_sequentially": True,
    "overwrite_raw_data": False,
    "generate_justification": True,
}
"============================================================"


def extract_json_from_response(response_text: str) -> dict:
    """
    Extracts and parses a JSON object from a model response string.

    Tries direct parsing first, then falls back to extracting the first
    {...} block via regex. Raises ValueError if neither approach succeeds.

    Arguments:
        • response_text:
            Raw text output from the model.

    Returns:
        • Parsed dict from the JSON object in the response.
    """
    stripped_response = response_text.strip()

    "Try direct parse"
    try:
        return json.loads(stripped_response)
    except json.JSONDecodeError:
        pass

    "Try to extract the first {...} block"
    json_block_pattern = re.search(r"\{[^{}]*\}", stripped_response, re.DOTALL)
    if json_block_pattern:
        try:
            return json.loads(json_block_pattern.group())
        except json.JSONDecodeError:
            pass

    "Try to find a larger nested JSON block"
    nested_json_pattern = re.search(r"\{.*\}", stripped_response, re.DOTALL)
    if nested_json_pattern:
        try:
            return json.loads(nested_json_pattern.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from model response: {response_text[:200]!r}")


def clamp_blame_rating(raw_value: Any) -> str:
    """
    Validates and returns a blame/wrongness rating as a plain integer string.

    Arguments:
        • raw_value:
            The value from the parsed JSON response (may be int, float, or str).

    Returns:
        • An integer string in [1, 9], or the original string if conversion fails.
    """
    try:
        numeric_value = int(float(str(raw_value)))
        clamped_value = max(1, min(9, numeric_value))
        return str(clamped_value)
    except (ValueError, TypeError):
        return str(raw_value)


def clamp_punishment_rating(raw_value: Any) -> str:
    """
    Validates and returns a punishment rating as a plain integer string.

    Arguments:
        • raw_value:
            The value from the parsed JSON response (may be int, float, or str).

    Returns:
        • An integer string in [0, 50], or the original string if conversion fails.
    """
    try:
        numeric_value = int(float(str(raw_value)))
        clamped_value = max(0, min(50, numeric_value))
        return str(clamped_value)
    except (ValueError, TypeError):
        return str(raw_value)


def format_turn_label_for_display(turn_label: str) -> str:
    """
    Converts an internal turn_label string into a human-readable display string.

    Arguments:
        • turn_label:
            Internal label such as "distal_firework_cc" or "proximate_trolley_ch".

    Returns:
        • A readable string such as "DISTAL  firework / CC" or "PROXIMATE  trolley / CH".
    """
    turn_label_display_map = {
        "cog_load_recall": "COGNITIVE LOAD RECALL",
        "comprehension": "COMPREHENSION CHECKS",
        "two_afc": "2AFC",
        "crt": "CRT",
        "indcol": "INDCOL",
    }

    if turn_label in turn_label_display_map:
        return turn_label_display_map[turn_label]

    parts = turn_label.split("_")
    if len(parts) >= 3:
        agent_type = parts[0].upper()
        story = parts[1]
        condition = parts[2].upper()
        return f"{agent_type:<10} {story} / {condition}"

    return turn_label.upper()


def summarize_ratings_for_display(parsed_response: dict, turn_label: str) -> str:
    """
    Formats parsed ratings into a compact one-line display string.

    Arguments:
        • parsed_response:
            The parsed JSON dict from the model response.
        • turn_label:
            Internal turn label used to determine which keys to show.

    Returns:
        • A compact summary string, e.g. "blame=7  wrongness=6  punishment=3".
    """
    if "blame" in parsed_response or "wrongness" in parsed_response or "punishment" in parsed_response:
        blame = parsed_response.get("blame", "?")
        wrongness = parsed_response.get("wrongness", "?")
        punishment = parsed_response.get("punishment", "?")
        return f"blame={blame}  wrongness={wrongness}  punishment={punishment}"

    if "recalled_number" in parsed_response:
        return f"recalled={parsed_response['recalled_number']!r}"

    if "q1" in parsed_response and turn_label == "comprehension":
        answers = [f"q{i}={parsed_response.get(f'q{i}', '?')}" for i in range(1, 4)]
        return "  ".join(answers)

    if "q1" in parsed_response and turn_label == "two_afc":
        return f"q1={parsed_response.get('q1', '?')[:40]}..."

    if "bat_ball" in parsed_response:
        return f"bat_ball={parsed_response.get('bat_ball')}  widgets={parsed_response.get('widgets')}  lily_pads={parsed_response.get('lily_pads')}"

    return str(parsed_response)[:80]


async def request_ratings_with_retry(
    model_client: ModelClient,
    conversation_history: list[dict],
    prompt_text: str,
    json_schema_description: str,
    beta_mode: bool,
    participant_id: str,
    turn_label: str,
) -> tuple[dict, list[dict]]:
    """
    Sends a prompt to the model and parses a JSON response, retrying on parse failure
    or rate limit errors (429) with exponential backoff.

    Appends the new user message and the model's assistant reply to conversation_history
    in place so subsequent turns have full context.

    Arguments:
        • model_client:
            The ModelClient instance to use for this participant.
        • conversation_history:
            Running list of message dicts for this participant's conversation.
        • prompt_text:
            The user-facing prompt for this turn.
        • json_schema_description:
            A description of the JSON format expected (included in the prompt).
        • beta_mode:
            If True, prints a summary line for every turn and full responses for ratings.
        • participant_id:
            Short ID string used in debug output.
        • turn_label:
            Short label for this turn used in debug output.

    Returns:
        • A tuple of (parsed_json_dict, updated_conversation_history).
    """
    full_prompt = f"{prompt_text}\n\nRespond with ONLY a JSON object in this exact format (no explanation, no markdown fences):\n{json_schema_description}"
    conversation_history.append({"role": "user", "content": full_prompt})

    turn_display = format_turn_label_for_display(turn_label)
    max_rate_limit_retries = 4
    rate_limit_wait_seconds = 15

    for rate_limit_attempt in range(max_rate_limit_retries):
        try:
            for parse_attempt in range(2):
                response_text = await model_client.chat(conversation_history)

                try:
                    parsed_response = extract_json_from_response(response_text)
                    conversation_history.append({"role": "assistant", "content": response_text})

                    if beta_mode:
                        ratings_summary = summarize_ratings_for_display(parsed_response, turn_label)
                        print(f"  [{participant_id}]  {turn_display}  →  {ratings_summary}")

                    return parsed_response, conversation_history

                except ValueError:
                    if parse_attempt == 0:
                        retry_prompt = "Your response could not be parsed as valid JSON. Please respond again with ONLY the JSON object, no other text."
                        conversation_history.append({"role": "assistant", "content": response_text})
                        conversation_history.append({"role": "user", "content": retry_prompt})
                    else:
                        print(f"  [{participant_id}]  {turn_display}  →  JSON parse failed after retry")
                        conversation_history.append({"role": "assistant", "content": response_text})
                        return {}, conversation_history

        except Exception as api_error:
            error_string = str(api_error).lower()
            is_rate_limit_error = "rate_limit" in error_string or "429" in error_string or "rate limit" in error_string

            if is_rate_limit_error and rate_limit_attempt < max_rate_limit_retries - 1:
                wait_time = rate_limit_wait_seconds * (2 ** rate_limit_attempt)
                print(f"  [{participant_id}]  {turn_display}  →  rate limit hit, waiting {wait_time}s before retry {rate_limit_attempt + 2}/{max_rate_limit_retries}")
                await asyncio.sleep(wait_time)
            else:
                raise

    return {}, conversation_history


async def run_single_participant(
    model_client: ModelClient,
    story_condition: str,
    load_condition: str,
    vignette_order: list[str],
    participant_id: str,
    robot_model_name_string: str,
    participant_temperature: float,
    generate_justification: bool,
    beta_mode: bool,
    print_transcripts: bool,
) -> dict:
    """
    Runs one robot participant through the full experiment protocol.

    The conversation mirrors the Qualtrics page sequence:
        1. Welcome + instructions
        2. Cognitive load assignment
        3. Vignette block instructions
        4-6. Three distal-agent vignettes (randomized order), each with blame/wrongness/punish
        7-8. Two proximate-agent vignettes (cc then ch), each with blame/wrongness/punish
        9. Cognitive load recall
        10. Comprehension checks
        11. 2AFC block
        12. CRT
        13. INDCOL

    Arguments:
        • model_client:
            The ModelClient instance to use for API calls.
        • story_condition:
            "firework" or "trolley".
        • load_condition:
            "high" or "low".
        • vignette_order:
            A list of three condition strings, e.g. ["cc", "div", "ch"].
        • participant_id:
            Unique identifier string for this participant.
        • robot_model_name_string:
            The model name string (e.g. "claude-sonnet-4-6"), stored in the model_name column.
        • beta_mode:
            If True, prints full transcript to terminal.
        • print_transcripts:
            If True and beta_mode is True, prints the full conversation.

    Returns:
        • A dict of column_name → value pairs matching the human raw CSV format.
    """
    start_time = time.time()
    start_datetime = datetime.now().strftime("%m/%d/%Y %H:%M")

    "Initialize the conversation with a system prompt that sets participant role without revealing hypotheses"
    conversation_history = [
        {
            "role": "system",
            "content": (
                "You are participating in a psychology research study. "
                "You will read several short scenarios and answer questions about the people involved. "
                "Please provide honest, thoughtful responses based on your moral intuitions. "
                "Follow the instructions exactly as given and do not skip any questions. "
                "When asked for numerical ratings, always provide a number within the requested range."
            ),
        }
    ]

    "Initialize the output row with all expected columns set to empty string"
    output_row = {column_name: "" for column_name in get_all_output_columns()}
    output_row["ResponseId"] = participant_id
    output_row["StartDate"] = start_datetime
    output_row["Progress"] = "100"
    output_row["Finished"] = "True"
    output_row["StoryCondition"] = story_condition
    output_row["LoadCondition"] = load_condition
    output_row["CaseOrder"] = "-".join([condition.upper() for condition in vignette_order])
    output_row["UserLanguage"] = "EN"
    output_row["Consent Form"] = "I CONSENT to participate in this study"
    output_row["Q_RecaptchaScore"] = "1"
    output_row["Age"] = "ROBOT_PARTICIPANT"
    output_row["Gender"] = "ROBOT_PARTICIPANT"
    output_row["Race"] = robot_model_name_string
    output_row["Race_7_TEXT"] = str(participant_temperature)
    output_row["Political"] = "ROBOT_PARTICIPANT"
    output_row["id"] = participant_id

    "Turn 1: Welcome and study instructions"
    conversation_history.append({"role": "user", "content": PARTICIPANT_INSTRUCTIONS + "\n\nPlease acknowledge that you understand and are ready to proceed by responding with: {\"ready\": true}"})
    acknowledgment_response = await model_client.chat(conversation_history)
    conversation_history.append({"role": "assistant", "content": acknowledgment_response})

    "Turn 2: Cognitive load assignment"
    cognitive_load_digit = COGNITIVE_LOAD_HIGH_DIGIT if load_condition == "high" else COGNITIVE_LOAD_LOW_DIGIT
    cognitive_load_instruction = COGNITIVE_LOAD_INSTRUCTION_TEMPLATE.format(digit=cognitive_load_digit)
    conversation_history.append({"role": "user", "content": cognitive_load_instruction + "\n\nAcknowledge by responding with: {\"digit_memorized\": true}"})
    load_acknowledgment = await model_client.chat(conversation_history)
    conversation_history.append({"role": "assistant", "content": load_acknowledgment})

    "Turn 3: Vignette block instructions"
    conversation_history.append({"role": "user", "content": VIGNETTE_BLOCK_INSTRUCTIONS + "\n\nAcknowledge by responding with: {\"instructions_understood\": true}"})
    instructions_acknowledgment = await model_client.chat(conversation_history)
    conversation_history.append({"role": "assistant", "content": instructions_acknowledgment})

    "Turns 4-6: Distal-agent vignettes in randomized order"
    for vignette_condition in vignette_order:
        vignette_text = VIGNETTE_TEXT[(story_condition, vignette_condition)]
        question_key = distal_question_key(story_condition, vignette_condition)

        blame_question = DISTAL_BLAME_QUESTION[question_key]
        wrong_question = DISTAL_WRONG_QUESTION[question_key]
        punish_question = DISTAL_PUNISH_QUESTION[question_key]

        vignette_prompt = (
            f"{vignette_text}\n\n"
            f"Now please answer the following three questions about Clark:\n\n"
            f"1. {blame_question}\n\n"
            f"2. {wrong_question}\n\n"
            f"3. {punish_question}"
        )

        json_schema = '{"blame": <integer 1-9>, "wrongness": <integer 1-9>, "punishment": <integer 0-50>}'
        column_prefix = DISTAL_COLUMN_PREFIX[(story_condition, vignette_condition)]
        turn_label = f"distal_{story_condition}_{vignette_condition}"

        parsed_ratings, conversation_history = await request_ratings_with_retry(
            model_client=model_client,
            conversation_history=conversation_history,
            prompt_text=vignette_prompt,
            json_schema_description=json_schema,
            beta_mode=beta_mode and print_transcripts,
            participant_id=participant_id,
            turn_label=turn_label,
        )

        output_row[f"{column_prefix}_blame_1"] = clamp_blame_rating(parsed_ratings.get("blame", ""))
        output_row[f"{column_prefix}_wrong_1"] = clamp_blame_rating(parsed_ratings.get("wrongness", ""))
        output_row[f"{column_prefix}_punish"] = clamp_punishment_rating(parsed_ratings.get("punishment", ""))

    "Turns 7-8: Proximate-agent vignettes (cc then ch, fixed order as in original study)"
    for proximate_condition in ["cc", "ch"]:
        preamble = PROXIMATE_PREAMBLE[(story_condition, proximate_condition)]
        vignette_text = VIGNETTE_TEXT[(story_condition, proximate_condition)]
        blame_question = PROXIMATE_BLAME_QUESTION[(story_condition, proximate_condition)]
        wrong_question = PROXIMATE_WRONG_QUESTION[(story_condition, proximate_condition)]
        punish_question = PROXIMATE_PUNISH_QUESTION[(story_condition, proximate_condition)]

        proximate_prompt = (
            f"{preamble}\n\n"
            f"{vignette_text}\n\n"
            f"Now please answer the following three questions:\n\n"
            f"1. {blame_question}\n\n"
            f"2. {wrong_question}\n\n"
            f"3. {punish_question}"
        )

        json_schema = '{"blame": <integer 1-9>, "wrongness": <integer 1-9>, "punishment": <integer 0-50>}'
        column_prefix = PROXIMATE_COLUMN_PREFIX[(story_condition, proximate_condition)]
        turn_label = f"proximate_{story_condition}_{proximate_condition}"

        parsed_ratings, conversation_history = await request_ratings_with_retry(
            model_client=model_client,
            conversation_history=conversation_history,
            prompt_text=proximate_prompt,
            json_schema_description=json_schema,
            beta_mode=beta_mode and print_transcripts,
            participant_id=participant_id,
            turn_label=turn_label,
        )

        output_row[f"{column_prefix}_blame_1"] = clamp_blame_rating(parsed_ratings.get("blame", ""))
        output_row[f"{column_prefix}_wrong_1"] = clamp_blame_rating(parsed_ratings.get("wrongness", ""))
        output_row[f"{column_prefix}_punish"] = clamp_punishment_rating(parsed_ratings.get("punishment", ""))

    "Turn 9: Cognitive load recall"
    parsed_recall, conversation_history = await request_ratings_with_retry(
        model_client=model_client,
        conversation_history=conversation_history,
        prompt_text=COGNITIVE_LOAD_RECALL_PROMPT,
        json_schema_description='{"recalled_number": "<the number you memorized>"}',
        beta_mode=beta_mode and print_transcripts,
        participant_id=participant_id,
        turn_label="cog_load_recall",
    )
    output_row["cog_load_check"] = str(parsed_recall.get("recalled_number", ""))

    "Turn 10: Comprehension checks"
    story_comprehension_questions = COMPREHENSION_QUESTIONS[story_condition]
    comp_prefix = COMPREHENSION_COLUMN_PREFIX[story_condition]

    comprehension_prompt = (
        "Please answer the following true-false comprehension questions about the stories you read.\n\n"
        f"1. True or False: {story_comprehension_questions['prob_harm']['text']}\n\n"
        f"2. True or False: {story_comprehension_questions['agency_across']['text']}\n\n"
        f"3. True or False: {story_comprehension_questions['agency_within']['text']}"
    )
    json_schema = '{"q1": "True" or "False", "q2": "True" or "False", "q3": "True" or "False"}'

    parsed_comprehension, conversation_history = await request_ratings_with_retry(
        model_client=model_client,
        conversation_history=conversation_history,
        prompt_text=comprehension_prompt,
        json_schema_description=json_schema,
        beta_mode=beta_mode and print_transcripts,
        participant_id=participant_id,
        turn_label="comprehension",
    )

    output_row[f"{comp_prefix}_prob_harm"] = str(parsed_comprehension.get("q1", ""))
    output_row[f"{comp_prefix}_agency_across"] = str(parsed_comprehension.get("q2", ""))
    output_row[f"{comp_prefix}_agency_within"] = str(parsed_comprehension.get("q3", ""))

    "Turn 11: 2AFC block"
    story_two_afc_questions = TWO_AFC_QUESTIONS[story_condition]
    afc_prefix = TWO_AFC_COLUMN_PREFIX[story_condition]

    interperson_primary_choices = story_two_afc_questions["interperson_primary"]
    interperson_followup_choices = story_two_afc_questions["interperson_followup"]
    ch_cc_primary_choices = story_two_afc_questions["intraperson_ch_cc_primary"]
    ch_cc_followup_choices = story_two_afc_questions["intraperson_ch_cc_followup"]
    div_cc_primary_choices = story_two_afc_questions["intraperson_div_cc_primary"]
    div_cc_followup_choices = story_two_afc_questions["intraperson_div_cc_followup"]

    two_afc_prompt = (
        "Please answer the following questions. For each question, you MUST choose EXACTLY ONE of the listed options "
        "and copy that option's text verbatim into your JSON response.\n\n"

        f"QUESTION 1: {interperson_primary_choices['text']}\n"
        f"  Option A: {interperson_primary_choices['choice_a']}\n"
        f"  Option B: {interperson_primary_choices['choice_b']}\n\n"

        f"QUESTION 2: {interperson_followup_choices['text']}\n"
        f"  Option A: {interperson_followup_choices['choice_a']}\n"
        f"  Option B: {interperson_followup_choices['choice_b']}\n\n"

        f"QUESTION 3: {ch_cc_primary_choices['text']}\n"
        f"  Option A: {ch_cc_primary_choices['choice_a']}\n"
        f"  Option B: {ch_cc_primary_choices['choice_b']}\n\n"

        f"QUESTION 4: {ch_cc_followup_choices['text']}\n"
        f"  Option A: {ch_cc_followup_choices['choice_a']}\n"
        f"  Option B: {ch_cc_followup_choices['choice_b']}\n\n"

        f"QUESTION 5: {div_cc_primary_choices['text']}\n"
        f"  Option A: {div_cc_primary_choices['choice_a']}\n"
        f"  Option B: {div_cc_primary_choices['choice_b']}\n\n"

        f"QUESTION 6: {div_cc_followup_choices['text']}\n"
        f"  Option A: {div_cc_followup_choices['choice_a']}\n"
        f"  Option B: {div_cc_followup_choices['choice_b']}"
    )

    json_schema = (
        '{"q1": "<exact text of chosen option>", "q2": "<exact text of chosen option>", '
        '"q3": "<exact text of chosen option>", "q4": "<exact text of chosen option>", '
        '"q5": "<exact text of chosen option>", "q6": "<exact text of chosen option>"}'
    )

    parsed_two_afc, conversation_history = await request_ratings_with_retry(
        model_client=model_client,
        conversation_history=conversation_history,
        prompt_text=two_afc_prompt,
        json_schema_description=json_schema,
        beta_mode=beta_mode and print_transcripts,
        participant_id=participant_id,
        turn_label="two_afc",
    )

    output_row[f"{afc_prefix}_interperson_1"] = str(parsed_two_afc.get("q1", ""))
    output_row[f"{afc_prefix}_interperson_2"] = str(parsed_two_afc.get("q2", ""))
    output_row[f"{afc_prefix}_intraperson_1"] = str(parsed_two_afc.get("q3", ""))
    output_row[f"{afc_prefix}_intraperson_2"] = str(parsed_two_afc.get("q4", ""))
    output_row[f"{afc_prefix}_intraperson_3"] = str(parsed_two_afc.get("q5", ""))
    output_row[f"{afc_prefix}_intraperson_4"] = str(parsed_two_afc.get("q6", ""))

    "Turn 12: CRT"
    crt_prompt = (
        "The following are short reasoning problems. Please read each problem carefully and give your best answer.\n\n"
        + "\n\n".join(
            f"{index + 1}. {item['text']}" for index, item in enumerate(CRT_QUESTIONS)
        )
    )
    json_schema = '{"bat_ball": <number>, "widgets": <number>, "lily_pads": <number>}'

    parsed_crt, conversation_history = await request_ratings_with_retry(
        model_client=model_client,
        conversation_history=conversation_history,
        prompt_text=crt_prompt,
        json_schema_description=json_schema,
        beta_mode=beta_mode and print_transcripts,
        participant_id=participant_id,
        turn_label="crt",
    )

    output_row["crt_bat_ball"] = str(parsed_crt.get("bat_ball", ""))
    output_row["crt_widgets"] = str(parsed_crt.get("widgets", ""))
    output_row["crt_lilly_pads"] = str(parsed_crt.get("lily_pads", ""))

    "Turn 13: INDCOL"
    indcol_prompt = (
        "Below are statements about attitudes, preferences, and social tendencies. "
        "Please indicate how true each statement is for you on a scale from 1 (totally disagree) to 9 (totally agree).\n\n"
        + "\n".join(
            f"{index + 1}. [{item['column']}] {item['text']}" for index, item in enumerate(INDCOL_ITEMS)
        )
    )
    indcol_json_keys = ", ".join(f'"{item["column"]}": <integer 1-9>' for item in INDCOL_ITEMS)
    json_schema = "{" + indcol_json_keys + "}"

    parsed_indcol, conversation_history = await request_ratings_with_retry(
        model_client=model_client,
        conversation_history=conversation_history,
        prompt_text=indcol_prompt,
        json_schema_description=json_schema,
        beta_mode=beta_mode and print_transcripts,
        participant_id=participant_id,
        turn_label="indcol",
    )

    for indcol_item in INDCOL_ITEMS:
        output_row[indcol_item["column"]] = clamp_blame_rating(parsed_indcol.get(indcol_item["column"], ""))

    "Optional free-text justification turn — stored in user_feedback"
    if generate_justification:
        justification_prompt = (
            "Finally, in exactly 2-3 sentences, explain what most influenced your ratings of Clark's "
            "blameworthiness across the three scenarios. Did the presence of another decision-maker "
            "between Clark's action and the outcome affect how responsible you felt Clark was? Why or why not?"
        )
        conversation_history.append({"role": "user", "content": justification_prompt})
        justification_response = await model_client.chat(conversation_history)
        conversation_history.append({"role": "assistant", "content": justification_response})
        output_row["user_feedback"] = justification_response.strip()

    "Finalize timing metadata"
    end_time = time.time()
    duration_seconds = int(end_time - start_time)
    end_datetime = datetime.now().strftime("%m/%d/%Y %H:%M")

    output_row["EndDate"] = end_datetime
    output_row["Duration (in seconds)"] = str(duration_seconds)

    return output_row


def get_all_output_columns() -> list[str]:
    """
    Returns the full list of output CSV column names in the same order as the human raw data,
    plus a robot-only model_name column at the end.

    Returns:
        • List of column name strings.
    """
    return [
        "StartDate", "EndDate", "Progress", "Duration (in seconds)", "Finished",
        "RecordedDate", "ResponseId", "UserLanguage", "Q_RecaptchaScore", "Consent Form",
        "pucc_blame_1", "pucc_wrong_1", "pucc_punish",
        "puch_blame_1", "puch_wrong_1", "puch_punish",
        "pudiv_blame_1", "pudiv_wrong_1", "pudiv_punish",
        "ppcc_blame_1", "ppcc_wrong_1", "ppcc_punish",
        "ppch_blame_1", "ppch_wrong_1", "ppch_punish",
        "tucc_blame_1", "tucc_wrong_1", "tucc_punish",
        "tuch_blame_1", "tuch_wrong_1", "tuch_punish",
        "tudiv_blame_1", "tudiv_wrong_1", "tudiv_punish",
        "tpcc_blame_1", "tpcc_wrong_1", "tpcc_punish",
        "tpch_blame_1", "tpch_wrong_1", "tpch_punish",
        "cog_load_check",
        "comp_p_prob_harm", "comp_p_agency_across", "comp_p_agency_within",
        "2afc_p_interperson_1", "2afc_p_interperson_2",
        "2afc_p_intraperson_1", "2afc_p_intraperson_2",
        "2afc_p_intraperson_3", "2afc_p_intraperson_4",
        "comp_t_prob_harm", "comp_t_agency_across", "comp_t_agency_within",
        "2afc_t_interperson_1", "2afc_t_interperson_2",
        "2afc_t_intraperson_1", "2afc_t_intraperson_2",
        "2afc_t_intraperson_3", "2afc_t_intraperson_4",
        "crt_bat_ball", "crt_widgets", "crt_lilly_pads",
        "indcol_hi_1_1", "indcol_hi_2_1", "indcol_hi_3_1",
        "indcol_vi_1_1", "indcol_vi_2_1", "indcol_vi_3_1", "indcol_vi_4_1",
        "indcol_hc_1_1", "indcol_hc_2_1", "indcol_hc_3_1", "indcol_hc_4_1",
        "indcol_vc_1_1", "indcol_vc_2_1", "indcol_vc_3_1",
        "Age", "Gender", "Race", "Race_7_TEXT", "Political", "Political_6_TEXT",
        "user_feedback", "id", "LoadCondition", "StoryCondition", "CaseOrder",
    ]


def generate_participant_conditions(
    n_participants: int, story_balance: str, random_seed: Optional[int]
) -> list[dict]:
    """
    Generates randomized condition assignments for each participant.

    Mirrors the between-subjects design of the human study:
        - story_condition: randomly "firework" or "trolley"
        - load_condition: randomly "high" or "low"
        - vignette_order: a random permutation of ["cc", "ch", "div"]

    Arguments:
        • n_participants:
            Number of participants to generate conditions for.
        • story_balance:
            "random" assigns story family independently per participant.
            "balanced" forces equal numbers of firework and trolley participants.
        • random_seed:
            Integer seed for reproducibility. None means unseeded (true variance).

    Returns:
        • List of dicts, each with keys: story_condition, load_condition, vignette_order, participant_id.
    """
    if random_seed is not None:
        random.seed(random_seed)

    all_vignette_orders = [
        ["cc", "ch", "div"], ["cc", "div", "ch"],
        ["ch", "cc", "div"], ["ch", "div", "cc"],
        ["div", "cc", "ch"], ["div", "ch", "cc"],
    ]

    participant_conditions_list = []

    for participant_index in range(n_participants):
        if story_balance == "balanced":
            story_condition_value = "firework" if participant_index % 2 == 0 else "trolley"
        else:
            story_condition_value = random.choice(["firework", "trolley"])

        load_condition_value = random.choice(["high", "low"])
        vignette_order_value = random.choice(all_vignette_orders).copy()
        participant_id_value = f"ROBOT_{uuid.uuid4().hex[:8].upper()}"

        participant_conditions_list.append({
            "story_condition": story_condition_value,
            "load_condition": load_condition_value,
            "vignette_order": vignette_order_value,
            "participant_id": participant_id_value,
        })

    return participant_conditions_list


def build_robot_general_settings(robot_raw_csv_path: Path) -> GeneralSettings:
    """
    Builds a robot-specific GeneralSettings dict for the analysis pipeline.

    Points all file paths at robot_raw_data/processed/ and robot_raw_data/tables/,
    bypasses the human freeze-window filter, and forces rebuild of all processed outputs
    so the analysis always reflects the current state of the raw data file.

    Arguments:
        • robot_raw_csv_path:
            Path to the robot raw data CSV file.

    Returns:
        • A GeneralSettings dict compatible with preprocessing.py, core.py, and tables.py.
    """
    robot_raw_data_folder_path = robot_raw_csv_path.parent
    robot_processed_folder_path = robot_raw_data_folder_path / "processed"
    robot_tables_folder_path = robot_raw_data_folder_path / "tables"

    robot_processed_folder_path.mkdir(parents=True, exist_ok=True)
    robot_tables_folder_path.mkdir(parents=True, exist_ok=True)

    return {
        "filing": {
            "file_paths": {
                "raw_data": robot_raw_data_folder_path,
                "processed": robot_processed_folder_path,
                "visuals": ROOT / "visuals",
                "images": ROOT / "images",
                "tables": robot_tables_folder_path,
                "root": ROOT,
            },
            "file_names": {
                "tests": "robot_responsibility_shielding_tests.csv",
                "cleaned": "robot_responsibility_shielding_cleaned.csv",
                "raw_data": robot_raw_csv_path.name,
                "group_summaries": "robot_responsibility_shielding_group_summaries.csv",
                "consistency_effects": "robot_responsibility_shielding_consistency_effects.csv",
                "afc_counts_table": "robot_responsibility_shielding_2afc_counts_table.csv",
                "afc_counts_long": "robot_responsibility_shielding_2afc_counts_long.csv",
                "triangulation": "robot_responsibility_shielding_triangulation.csv",
                "correlations": "robot_responsibility_shielding_correlations.csv",
                "regressions": "robot_responsibility_shielding_regressions.csv",
                "first_vignette": "robot_responsibility_shielding_integrated_first_vignette_blame_models.csv",
                "within_subject": "robot_responsibility_shielding_integrated_within_subject_blame_models.csv",
                "blame_models": "robot_responsibility_shielding_integrated_blame_models.csv",
                "codebook": "robot_responsibility_shielding_processed_codebook.csv",
            },
            "table_names": {
                "table_1_participant_counts": "robot_Table_1_Participant_Counts.csv",
                "table_2_means_by_dv_and_condition": "robot_Table_2_Means_by_DV_and_Condition.csv",
                "table_3_primary_distal_blame_contrasts": "robot_table_3_Primary_Distal_Blame_Contrasts.csv",
                "table_4_story_specific_distal_blame_contrasts": "robot_table_4_Story_Specific_Distal_Blame_Contrasts.csv",
                "table_5_two_alternative_forced_choice_distribution": "robot_table_5_Two_Alternative_Forced_Choice_Distribution.csv",
                "table_6_within_subject_pairwise_blame_matrix": "robot_table_6_Within_Subject_Pairwise_Blame_Matrix.csv",
                "table_6_within_subject_pairwise_blame_long": "robot_table_6_Within_Subject_Pairwise_Blame_Long.csv",
                "table_7_cognitive_load_blame_contrasts": "robot_table_7_Cognitive_Load_Blame_Contrasts.csv",
                "table_8_order_effects_summary": "robot_table_8_Order_Effects_Summary.csv",
                "table_9_secondary_dv_contrasts": "robot_table_9_Secondary_DV_Contrasts.csv",
                "table_manifest": "robot_Table_Manifest.csv",
            },
        },
        "visuals": {
            "figure_layout": {},
            "marker_size": 7,
            "create_figures": False,
            "export_figure": False,
            "dark_mode": False,
            "base_hue": 220,
        },
        "punish": {
            "analysis_mode": "log1p_parametric",
            "bootstrap_iterations": 1000,
            "random_seed": 2026,
        },
        "misc": {
            "confirmatory_between_subjects_method": "pooled_ols",
            "rebuild_cleaned_dataframe": True,
            "print_tables_to_terminal": True,
            "freeze_timestamp_first": "2/19/2026 10:57:56 PM",
            "freeze_timestamp_last": "3/20/2026 10:00:09 AM",
            "use_integrated_models": False,
            "force_rebuild": True,
            "one_tailed": True,
            "skip_freeze_filter": True,
        },
        "robot": {},
    }


def run_robot_analysis(robot_raw_csv_path: Path) -> None:
    """
    Runs the core analysis pipeline on robot participant data and prints Tables 2, 3, 4, and 9.

    Calls the same functions used in analysis.py — load_or_build_cleaned_dataframe,
    compute_group_summaries, run_confirmatory_and_exploratory_tests, and the four table
    builders — without any modification. Robot analysis outputs go to robot_raw_data/processed/
    and robot_raw_data/tables/; the human processed/ and tables/ folders are never touched.

    Each table function call is wrapped in its own try/except so a failure in one step
    (e.g. too few participants to run a test) does not abort the rest of the analysis.

    Arguments:
        • robot_raw_csv_path:
            Path to the robot raw data CSV file to analyze.

    Returns:
        • None. Outputs are printed to the terminal and written to disk.
    """
    print(f"\n{'='*60}")
    print(f"Robot Analysis Pipeline")
    print(f"Input: {robot_raw_csv_path.name}")
    print(f"{'='*60}\n")

    robot_general_settings = build_robot_general_settings(robot_raw_csv_path)

    try:
        robot_cleaned_dataframe = load_or_build_cleaned_dataframe(
            general_settings=robot_general_settings,
            force_rebuild=True,
        )
        total_robot_participants = len(robot_cleaned_dataframe)
        included_robot_participants = int(robot_cleaned_dataframe["included"].sum())
        print(f"Participants: {total_robot_participants} total, {included_robot_participants} passed comprehension checks\n")
    except Exception as cleaning_error:
        print(f"ERROR in preprocessing step: {cleaning_error}")
        traceback.print_exc()
        return

    try:
        compute_group_summaries(
            general_settings=robot_general_settings,
            force_rebuild=True,
        )
    except Exception as summaries_error:
        print(f"WARNING: group summaries step failed: {summaries_error}")

    try:
        run_confirmatory_and_exploratory_tests(
            general_settings=robot_general_settings,
            cleaned_dataframe=robot_cleaned_dataframe,
            force_rebuild=True,
        )
    except Exception as tests_error:
        print(f"WARNING: confirmatory tests step failed: {tests_error}")

    try:
        robot_table_2 = compute_manuscript_table_2_mean_scale_values_by_dv_and_condition(
            general_settings=robot_general_settings,
            force_rebuild=True,
        )
        print(f"--- Table 2: Means by DV and Condition ---")
        print(robot_table_2.to_string())
        print()
    except Exception as table_2_error:
        print(f"WARNING: Table 2 failed: {table_2_error}")

    try:
        robot_table_3 = compute_manuscript_table_3_primary_distal_blame_contrasts(
            general_settings=robot_general_settings,
            cleaned_dataframe=robot_cleaned_dataframe,
            force_rebuild=True,
        )
        print(f"--- Table 3: Primary Distal Blame Contrasts ---")
        print(robot_table_3.to_string())
        print()
    except Exception as table_3_error:
        print(f"WARNING: Table 3 failed: {table_3_error}")

    try:
        robot_table_4 = compute_manuscript_table_4_story_specific_distal_blame_contrasts(
            general_settings=robot_general_settings,
            cleaned_dataframe=robot_cleaned_dataframe,
            force_rebuild=True,
        )
        print(f"--- Table 4: Story-Specific Distal Blame Contrasts ---")
        print(robot_table_4.to_string())
        print()
    except Exception as table_4_error:
        print(f"WARNING: Table 4 failed: {table_4_error}")

    try:
        robot_table_9 = compute_supplementary_table_9_secondary_dv_contrasts(
            general_settings=robot_general_settings,
            cleaned_dataframe=robot_cleaned_dataframe,
            force_rebuild=True,
        )
        print(f"--- Table 9: Secondary DV Contrasts ---")
        print(robot_table_9.to_string())
        print()
    except Exception as table_9_error:
        print(f"WARNING: Table 9 failed: {table_9_error}")

    print(f"Analysis outputs written to: {robot_general_settings['filing']['file_paths']['processed']}")
    print(f"Table outputs written to:    {robot_general_settings['filing']['file_paths']['tables']}\n")


def get_backup_file_path(output_file_path: Path) -> Path:
    """
    Returns the backup file path derived from the primary output CSV path.

    The backup name has an underscore appended to the stem before the extension,
    e.g. robot_responsibility_shielding_raw.csv → robot_responsibility_shielding_raw_.csv.
    This backup is written when the main file is locked (e.g. open in Excel).

    Arguments:
        • output_file_path:
            Path to the primary output CSV.

    Returns:
        • Path with underscore appended to the stem.
    """
    return output_file_path.with_name(output_file_path.stem + "_" + output_file_path.suffix)


def merge_backup_if_exists(
    output_file_path: Path,
    backup_file_path: Path,
    output_column_names: list[str],
) -> None:
    """
    Merges any leftover backup CSV into the main output file, then deletes the backup.

    Called at experiment startup. If a previous run hit a PermissionError (e.g. the
    main CSV was open in Excel), rows were written to the backup file instead. This
    function recovers those rows before the new run begins.

    Arguments:
        • output_file_path:
            The primary output CSV path.
        • backup_file_path:
            The backup CSV path (underscore variant).
        • output_column_names:
            Column name list used when writing the header if the main file is new.

    Returns:
        • None. Appends recovered rows to output_file_path and deletes backup_file_path.
    """
    if not backup_file_path.exists():
        return

    print(f"\nFound backup file from a previous run where the CSV was locked: {backup_file_path.name}")

    try:
        with open(backup_file_path, newline="", encoding="utf-8") as backup_file:
            backup_rows = list(csv.DictReader(backup_file))

        if not backup_rows:
            backup_file_path.unlink()
            print("Backup file was empty. Deleted.\n")
            return

        write_header = not output_file_path.exists() or output_file_path.stat().st_size == 0
        with open(output_file_path, mode="a", newline="", encoding="utf-8") as main_file:
            writer = csv.DictWriter(main_file, fieldnames=output_column_names, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for row in backup_rows:
                writer.writerow(row)

        backup_file_path.unlink()
        print(f"Merged {len(backup_rows)} rows from backup into {output_file_path.name}. Backup deleted.\n")

    except Exception as merge_error:
        print(f"WARNING: Could not merge backup ({merge_error}). Backup preserved at: {backup_file_path}\n")


async def run_experiment(config: RobotSettings) -> None:
    """
    Runs the full robot participant experiment based on ROBOT_EXPERIMENT_CONFIG.

    Iterates over config["models"], running each model's participants sequentially
    (if run_models_sequentially=True) or in one combined concurrent pool. Within
    each model, participants run concurrently up to max_concurrent_participants.
    All models write to the same output CSV file; the model_name column identifies
    which model each row came from.

    Raw data is appended to the existing CSV by default. To clear and start fresh,
    set overwrite_raw_data=True in the config.

    Arguments:
        • config:
            The ROBOT_EXPERIMENT_CONFIG dict defined at the top of this file.

    Returns:
        • None. Results are written to config["output_file"].
    """
    is_beta_mode = config.get("beta_mode", False)
    n_participants_per_model = config.get("beta_n_participants", 3) if is_beta_mode else config.get("n_participants_per_model", 10)
    print_transcripts = config.get("print_transcripts", True)
    models_to_run = config.get("models", ["claude-sonnet-4-6"])
    run_models_sequentially = config.get("run_models_sequentially", True)

    output_file_path = Path(config["output_file"])
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    backup_file_path = get_backup_file_path(output_file_path)

    "Clear the raw data file only if explicitly requested — otherwise append to preserve existing runs"
    if config.get("overwrite_raw_data", False):
        if output_file_path.exists():
            output_file_path.unlink()
            print(f"Cleared existing raw data file: {output_file_path.name}")
        if backup_file_path.exists():
            backup_file_path.unlink()
            print(f"Cleared backup file: {backup_file_path.name}")
    else:
        merge_backup_if_exists(output_file_path, backup_file_path, get_all_output_columns())

    print(f"\n{'='*60}")
    print(f"Robot Participant Experiment")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Participants per model: {n_participants_per_model}")
    print(f"Temperature: {config.get('temperature', 1.0)} | Beta mode: {is_beta_mode}")
    print(f"Overwrite raw data: {config.get('overwrite_raw_data', False)}")
    print(f"Output: {output_file_path}")
    print(f"{'='*60}\n")

    if is_beta_mode:
        print("BETA MODE: Running a small test run. Check transcripts and output CSV before committing to a full run.\n")

    output_column_names = get_all_output_columns()

    "In beta mode force concurrency to 1 to avoid rate limit errors on new API accounts"
    max_concurrent = 1 if is_beta_mode else config.get("max_concurrent_participants", 5)

    total_completed = 0
    total_failed = 0

    async def run_one_participant_with_semaphore(
        participant_conditions: dict,
        robot_model_client: ModelClient,
        robot_model_name_string: str,
        semaphore: asyncio.Semaphore,
        participant_temperature: float,
    ) -> Optional[dict]:
        async with semaphore:
            participant_id = participant_conditions["participant_id"]
            order_display = " → ".join(c.upper() for c in participant_conditions["vignette_order"])
            print(f"\n{'─'*60}")
            print(f"  Participant: {participant_id}")
            print(f"  Model: {robot_model_name_string}  |  Story: {participant_conditions['story_condition']}  |  Load: {participant_conditions['load_condition']}  |  Order: {order_display}")
            print(f"{'─'*60}")
            try:
                result_row = await run_single_participant(
                    model_client=robot_model_client,
                    story_condition=participant_conditions["story_condition"],
                    load_condition=participant_conditions["load_condition"],
                    vignette_order=participant_conditions["vignette_order"],
                    participant_id=participant_id,
                    robot_model_name_string=robot_model_name_string,
                    participant_temperature=participant_temperature,
                    generate_justification=config.get("generate_justification", False),
                    beta_mode=is_beta_mode,
                    print_transcripts=print_transcripts,
                )
                print(f"  [{participant_id}]  ✓ completed")
                return result_row
            except Exception as participant_error:
                print(f"  [{participant_id}]  ✗ ERROR: {participant_error}")
                traceback.print_exc()
                return None

    async def run_one_model(robot_model_name_string: str) -> None:
        nonlocal total_completed, total_failed

        print(f"\n{'='*60}")
        print(f"Running model: {robot_model_name_string}")
        print(f"{'='*60}")

        requested_temperature = config.get("temperature", 1.0)
        provider_name = MODEL_TO_PROVIDER.get(robot_model_name_string, "openai")
        max_temperature = PROVIDER_MAX_TEMPERATURE.get(provider_name, 2.0)
        effective_temperature = min(requested_temperature, max_temperature)
        if effective_temperature != requested_temperature:
            print(f"  Note: temperature clipped {requested_temperature} → {effective_temperature} (max for {provider_name})")

        robot_model_client = get_client_for_model(
            model_name=robot_model_name_string,
            temperature=effective_temperature,
        )

        participant_conditions_list = generate_participant_conditions(
            n_participants=n_participants_per_model,
            story_balance=config.get("story_balance", "random"),
            random_seed=None,
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        tasks = [
            run_one_participant_with_semaphore(conditions, robot_model_client, robot_model_name_string, semaphore, effective_temperature)
            for conditions in participant_conditions_list
        ]
        completed_rows = await asyncio.gather(*tasks)

        valid_rows = [row for row in completed_rows if row is not None]
        model_completed = len(valid_rows)
        model_failed = len(completed_rows) - model_completed

        def _write_rows(target_path: Path) -> None:
            write_header = not target_path.exists() or target_path.stat().st_size == 0
            with open(target_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=output_column_names, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                for row in valid_rows:
                    writer.writerow(row)

        try:
            _write_rows(output_file_path)
        except PermissionError:
            print(f"\n  WARNING: {output_file_path.name} is locked (probably open in Excel).")
            print(f"  Saving {model_completed} rows to backup: {backup_file_path.name}")
            print(f"  Close the CSV and re-run — the backup will be merged automatically.\n")
            _write_rows(backup_file_path)

        total_completed += model_completed
        total_failed += model_failed

        print(f"\n{robot_model_name_string}: {model_completed} completed, {model_failed} failed")

    if run_models_sequentially:
        for robot_model_name_string in models_to_run:
            await run_one_model(robot_model_name_string)
    else:
        await asyncio.gather(*[run_one_model(robot_model_name_string) for robot_model_name_string in models_to_run])

    print(f"\n{'='*60}")
    print(f"Experiment complete.")
    print(f"Total participants completed: {total_completed}")
    print(f"Total participants failed:    {total_failed}")
    print(f"Output written to: {output_file_path}")
    print(f"{'='*60}\n")

    if is_beta_mode:
        print("Beta run finished. Before running the full experiment:")
        print("  1. Check the output CSV columns match the human raw data format")
        print("  2. Inspect the ratings for plausibility")
        print("  3. Check that 2AFC responses exactly match the expected choice strings")
        print("  4. If everything looks good, set beta_mode=False and run again\n")

    if config.get("run_analysis_after_collection", True) and total_completed > 0:
        run_robot_analysis(output_file_path)


def main() -> None:
    """
    Entry point. Reads ROBOT_EXPERIMENT_CONFIG and runs the experiment.
    """
    asyncio.run(run_experiment(ROBOT_EXPERIMENT_CONFIG))


if __name__ == "__main__":
    main()
