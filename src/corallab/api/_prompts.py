"""Approximate the largest prompt each role will send, before a run.

The conversation does not exist yet, so a synthetic transcript of
``max_turns`` exchanges stands in for it, each message a fixed number of
tokens. Every prompt is built by the code that builds it in a real run
(the person generator, the fidelity checker, the judge's Stage 1 and Stage 2
builders) and counted with ``litellm.token_counter``, as ``--dry-run`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from corallab.api._manifests import ROLE_FIDELITY, ROLE_PERSON, ROLE_SUT
from corallab.schemas.core.messages import Turn
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.run_manifest import JudgeConfig, RunManifest

ROLE_JUDGE_STAGE1 = "judge_stage1"
ROLE_JUDGE_STAGE2 = "judge_stage2"


@dataclass(frozen=True)
class TranscriptAssumption:
    """Tokens per synthetic message. Real replies vary; these set the estimate's scale."""

    user_tokens: int = 60
    assistant_tokens: int = 350

    def describe(self) -> str:
        return f"~{self.user_tokens} tokens per user message, ~{self.assistant_tokens} per AI reply"


@dataclass(frozen=True)
class PromptSize:
    role: str
    tokens: int
    note: str


def _text(tokens: int) -> str:
    # " word" is one token in the common BPE vocabularies.
    return ("word " * max(tokens, 0)).strip()


def synthetic_turns(
    exchanges: int, assumption: TranscriptAssumption, *, final_reply: bool = True
) -> list[Turn]:
    """``exchanges`` user/assistant pairs; the last reply is left off when ``final_reply`` is
    False."""
    turns: list[Turn] = []
    for number in range(1, exchanges + 1):
        turns.append(Turn(turn_number=number, role="user", content=_text(assumption.user_tokens)))
        if final_reply or number < exchanges:
            turns.append(
                Turn(
                    turn_number=number,
                    role="assistant",
                    content=_text(assumption.assistant_tokens),
                )
            )
    return turns


def count_tokens(litellm_model: str, messages: list[dict[str, Any]]) -> int:
    import litellm

    return int(litellm.token_counter(model=litellm_model, messages=messages))


def _as_messages(turns: list[Turn]) -> list[dict[str, Any]]:
    return [{"role": t.role, "content": t.content} for t in turns]


def sut_prompt(
    manifest: RunManifest, cases: list[RunCase], litellm_model: str, a: TranscriptAssumption
) -> PromptSize:
    from corallab.simulation.setup.simulation_runner import scenario_content_for

    level = manifest.evaluation_level
    best = 0
    for case in cases:
        if level == 1:
            turns = [Turn(turn_number=1, role="user", content=case.test_case.turns[0])]
        elif level == 2:
            scripted = case.test_case.turns
            turns = []
            for n, text in enumerate(scripted, start=1):
                turns.append(Turn(turn_number=n, role="user", content=text))
                if n < len(scripted):
                    turns.append(
                        Turn(turn_number=n, role="assistant", content=_text(a.assistant_tokens))
                    )
        else:
            turns = synthetic_turns(manifest.simulation.max_turns, a, final_reply=False)
        messages = _as_messages(turns)
        system = scenario_content_for(case, manifest).get("target_system_prompt")
        if system:
            messages.insert(0, {"role": "system", "content": system})
        best = max(best, count_tokens(litellm_model, messages))
    note = (
        "the scripted turns"
        if level == 1
        else f"the last call of a {manifest.simulation.max_turns}-exchange conversation, "
        + a.describe()
    )
    return PromptSize(ROLE_SUT, best, note)


def person_prompt(
    manifest: RunManifest, cases: list[RunCase], litellm_model: str, a: TranscriptAssumption
) -> PromptSize:
    from corallab.simulation.person_response_generator import PersonResponseGenerator
    from corallab.simulation.setup.simulation_runner import scenario_content_for

    max_turns = manifest.simulation.max_turns
    history = synthetic_turns(max(max_turns - 1, 1), a)
    best = 0
    for case in cases:
        generator = PersonResponseGenerator(
            client=None,  # type: ignore[arg-type]  # builds prompts only, never calls
            persona=case.persona.rendering,
            scenario_content=scenario_content_for(case, manifest),
        )
        system = generator._build_system_prompt(turns_remaining=1)
        messages = [{"role": "system", "content": system}, *generator._build_messages(history)]
        best = max(best, count_tokens(litellm_model, messages))
    note = f"persona prompt plus {max(max_turns - 1, 1)} prior exchanges, " + a.describe()
    return PromptSize(ROLE_PERSON, best, note)


def fidelity_prompt(
    manifest: RunManifest, cases: list[RunCase], litellm_model: str, a: TranscriptAssumption
) -> PromptSize:
    from corallab.fidelity.elicitation import PROMPT_VERSION, _build_prompt

    version = manifest.simulation.elicitation_fidelity.prompt_version or PROMPT_VERSION
    turns = synthetic_turns(manifest.simulation.max_turns, a)
    best = 0
    for case in cases:
        prompt = _build_prompt(turns, case.test_case, version)
        best = max(best, count_tokens(litellm_model, [{"role": "user", "content": prompt}]))
    note = f"{manifest.simulation.max_turns}-exchange transcript, " + a.describe()
    return PromptSize(ROLE_FIDELITY, best, note)


def judge_prompts(
    manifest: RunManifest,
    judge: JudgeConfig,
    cases: list[RunCase],
    kg: Any,
    litellm_model: str,
    a: TranscriptAssumption,
) -> list[PromptSize]:
    """Largest Stage 1 prompt (exact apart from the transcript) and a Stage 2 lower bound."""
    from corallab.judging.judges.kl_two_stage import (
        KlTwoStageConfig,
        build_definition_groups,
        build_stage1_prompt,
        build_stage2_prompt,
        load_prompts,
    )

    if not cases:
        return []
    cfg = KlTwoStageConfig.from_options(judge.options)
    s1_template, s2_template, _hash = load_prompts(cfg.prompt_version)
    turns = synthetic_turns(_transcript_exchanges(manifest, cases), a)
    case = cases[0]
    persona = case.persona.rendering
    groups = [
        *build_definition_groups(kg, "ai_behavior", cfg.group_size),
        *build_definition_groups(kg, "user_behavior", cfg.group_size),
    ]

    def count(prompt: str) -> int:
        return count_tokens(litellm_model, [{"role": "user", "content": prompt}])

    stage1 = max(
        (count(build_stage1_prompt(s1_template, kg, g, turns, persona, cfg)[0]) for g in groups),
        default=0,
    )
    pathways = kg.pathways()
    size = cfg.pathways_per_call or max(1, len(pathways))
    batches = [pathways[i : i + size] for i in range(0, len(pathways), size)]
    stage2 = max(
        (
            count(build_stage2_prompt(s2_template, kg, b, [], turns, persona, cfg, case.facts)[0])
            for b in batches
        ),
        default=0,
    )
    transcript = f"{len(turns) // 2}-exchange transcript, " + a.describe()
    return [
        PromptSize(
            ROLE_JUDGE_STAGE1,
            stage1,
            f"largest of {len(groups)} definition groups at group_size {cfg.group_size}, "
            + transcript,
        ),
        PromptSize(
            ROLE_JUDGE_STAGE2,
            stage2,
            f"largest pathway batch (pathways_per_call {cfg.pathways_per_call}) without the "
            "Stage 1 timeline, so a lower bound; " + transcript,
        ),
    ]


def _transcript_exchanges(manifest: RunManifest, cases: list[RunCase]) -> int:
    if manifest.evaluation_level == 1:
        return 1
    if manifest.evaluation_level == 2:
        return max(len(c.test_case.turns) for c in cases)
    return manifest.simulation.max_turns
