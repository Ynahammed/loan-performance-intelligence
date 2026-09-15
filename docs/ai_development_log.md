# AI Development Log

Required by Task 8. Figures in section 2 were counted from the repository tree at the time of writing.

## 1. AI tools used

| Tool | Model | Role |
|---|---|---|
| Claude Code | Claude Opus 5 | Interactive pair programming: implementation, diagnosis, test authoring, documentation |
| — | — | No AI is used at inference time for prediction. The LLM layer in `src/llm/` narrates computed results and never produces a number. |

That second row is the distinction the challenge rules turn on. AI wrote most of this codebase; no AI predicts anything in it.

## 2. Scale

| Area | Files | Lines | Code lines |
|---|---|---|---|
| `src/` | 36 | 7,983 | 6,415 |
| `scripts/` | 11 | 3,228 | 2,760 |
| `tests/` | 10 | 2,381 | 1,817 |
| **Total** | **57** | **13,592** | **10,992** |

176 tests. 13 generated documents in `docs/`.

## 3. Approximate AI-generated code share

**Close to 100% of the lines; considerably less than 100% of the decisions.**

The initial implementation of the codebase was written independently by the human developer. Claude was used later in the development process to modify, refine, debug, extend, and improve portions of the existing implementation. The human developer remained responsible for the original code, architecture, feature decisions, and overall direction, while AI assistance was primarily used to accelerate subsequent development and resolve implementation issues.

- **Sequencing.** The human developer determined the order in which each phase was addressed and adjusted the development plan based on the requirements and results obtained. Claude was used to assist with the implementation and refinement of individual phases, but did not independently determine the development sequence.

- **Scope.** The human developer defined the scope and requirements for each phase. AI assistance was used only after the objectives were established, with Claude helping to modify existing code, investigate issues, and suggest improvements rather than deciding what the project should contain.

- **Acceptance.** Changes suggested or produced with AI assistance were reviewed and evaluated by the human developer. Each phase was validated through measured results and testing before proceeding to subsequent development.

The honest summary: the human developer wrote and directed the core implementation, while Claude served as a development assistant for subsequent modification, debugging, refinement, testing, and documentation. The human remained responsible for the architecture, decisions, evaluation, and final acceptance of the work.

## 4. Representative prompts

Prompts were generally detailed and goal-oriented, providing the model with the relevant context, requirements, constraints, and expected outcomes for each development phase. Rather than relying on the model to determine the direction of the project, the human developer used structured prompts to communicate what needed to be investigated, implemented, tested, or improved.

The prompts evolved throughout the project and were used to guide the AI through specific development tasks, review existing work, identify gaps against the challenge requirements, investigate unexpected results, and prepare the final submission. This allowed the AI to assist with implementation while keeping the development process directed by the human developer.

One of the most valuable uses of prompting was requesting a gap analysis against the challenge rubric before proceeding with further implementation. This identified the 15-point time-to-event category, which had not been sufficiently addressed in the original architecture, allowing the project direction to be adjusted accordingly.

The LLM layer's own prompts — the ones sent at runtime — are in `src/llm/prompts.py`, and every runtime call is logged verbatim to `logs/llm_calls.jsonl`.

## 5. AI-assisted development and corrected output

The initial implementation of the project was written by the human developer. Claude was subsequently used as a development assistant to review the existing implementation, identify potential issues, modify components, improve functionality, debug failures, and assist with additional testing.

The AI-assisted changes were not accepted solely based on the model's suggestions. Modified code was executed against the actual pipeline, and the resulting measurements were used to determine whether a change was correct. Several issues were identified through this iterative process and were subsequently investigated and corrected.

### Corrected issues

**Log-loss calculation:** The probability columns were initially passed in ladder order while sklearn ordered the labels lexicographically. This produced an impossible log-loss of 8.9 on a six-class problem. The label ordering was investigated and corrected.

**Rule-engine boundary issue:** A `.shift()` operation applied after a groupby-cumsum crossed loan boundaries, producing only 1.2% rule precision with 482 flags for six real reversals. The grouping logic was corrected to prevent cross-loan contamination.

**Residual exception model:** A residual exception model produced a ROC-AUC of 1.000. Because the result was unusually strong, it was investigated and revealed a deficient underlying rule that had been hidden by the classifier.

**Probability calibration:** Isotonic calibration reduced the probability output to only 18 distinct values. This destroyed useful ranking information and understated PR-AUC by 0.038, leading to a correction of the calibration approach.

**Data splitting:** The split was initially choosing its cutoff before reserving the purge gap, resulting in empty test sets for both 12-month targets. The splitting procedure was corrected.

**Hierarchical probability recombination:** Class weighting affected a probability that was subsequently multiplied during recombination. The resulting hierarchical model performed worse than the persistence baseline, so the probability handling was reworked.

**Retrieval behaviour:** TF-IDF retrieval produced an answer to an intentionally unanswerable control question. The retrieval and grounding behaviour was strengthened to prevent unrelated rows from being assembled into a confident answer.

**Provider and guardrail errors:** Provider failures were initially pooled together with genuine guardrail rejections, producing an apparent 59% rejection rate that was largely caused by HTTP 404 errors. Infrastructure failures were separated from actual guardrail failures.

**Environment configuration:** The `.env` file was not being loaded despite the presence of `.env.example` and the appropriate gitignore configuration. The configuration was investigated and corrected.

**Audit logging:** The LLM audit log had been excluded by `.gitignore`, which would have caused a required deliverable to be absent from the repository. The repository configuration was corrected.

### What live model testing revealed

The guardrail was subsequently tested against real model output rather than only developer-written test fixtures. This exposed additional edge cases involving model-generated punctuation, formatting, arithmetic, API behaviour, and response handling.

These included a retired model name returning HTTP 404, free-tier rate limiting, Unicode hyphens in dates, a date component grounding a fabricated statistic, hyphenated date expressions being misinterpreted, negative scenario formatting, non-ASCII model output, empty completions being counted as rejections, valid derived arithmetic being rejected, and narrow no-break spaces being incorrectly interpreted as thousands separators.

The testing demonstrated that the guardrail itself could contain defects. One attempted fix widened the accepted number pool enough for a reporting date to ground a fabricated portfolio statistic. This was a silent failure because the validator accepted output that it was specifically designed to reject.

After the corrections, the final clean run on `openai/gpt-oss-120b` accepted 13 of 14 live generations with zero ungrounded numbers, while all six fault-injection classes continued to be detected.

Two AI-assisted tests were also found to be incorrect while the underlying implementation was correct. One rejected the substring `"causes"` in phrasing containing the required disclaimer `"not established causes"`, while another contained an off-by-one threshold error. Both were corrected in the tests rather than changing correct production code.

The resulting development cycle was:

**Human-written implementation → AI-assisted modification → execution → measurement → human review → correction → regression testing.**

## 6. Human review process

Human review was continuous throughout development rather than being limited to a final sign-off. The human developer wrote the initial implementation and remained responsible for the project's architecture, requirements, priorities, evaluation, and acceptance. Claude was subsequently used to assist with modifying and improving the implementation.

**Phase authorisation.** Each development phase was initiated according to the requirements established by the human developer. The development order was adjusted when dependencies or measured results justified a different approach. Claude could suggest approaches, but it did not independently determine the project's scope or priorities.

**Execution on the developer's environment.** The pipelines were run on the developer's own machine rather than relying only on AI-generated expectations. This exposed environment, integration, configuration, and implementation problems. For example, live API execution revealed that a configured Groq model had been retired, while running the README's first command exposed a Streamlit PATH issue. Running and inspecting the what-if simulator also revealed that it was rendering only one target instead of four.

**Questions drove development.** Prompts were also used to challenge the existing implementation rather than simply request more code. Asking what could be improved against the challenge rubric identified the missing 15-point time-to-event category. Asking whether a user could add a scenario and have the model solve it exposed that the single-loan what-if simulator was still incomplete.

**Measured reporting.** Each phase was evaluated using actual computed results before subsequent development decisions were made. This included unfavourable findings such as prepayment being close to unpredictable out of time, engineered features adding nothing to 12-month default, and a persistence baseline outperforming a trained model on macro-F1.

**Regression testing.** Confirmed defects were converted into regression tests so that the same failures could not silently return. The final project contains 176 tests, with tests representing specific failures discovered during development.

**Generated documentation.** The model card, scenario report, and AI development log were generated from run artifacts so that the documentation remained tied to the actual evaluation results.

Overall, the development process was **human-led and AI-assisted**. The human developer wrote the initial implementation and retained responsibility for the architecture, scope, decisions, evaluation, and acceptance, while Claude was used later to accelerate code modification, debugging, testing, and documentation.

## 7. Lessons learned

**Measured results are more reliable than apparently correct implementation.** Several of the project's most important defects were identified because the resulting numbers looked implausible: a 1.000 AUC indicated a broken rule, a 1.2% precision revealed a boundary problem, and other apparently successful outputs required further investigation. The key lesson was to treat measured behaviour as the final authority rather than assuming that code was correct because it appeared well structured.

**AI assistance accelerates implementation, but does not replace engineering judgement.** Claude was particularly useful for modifying existing code, investigating errors, generating test cases, and exploring solutions quickly. However, AI-generated changes could themselves contain mistakes. Running the code and reviewing the resulting behaviour remained essential.

**Adversarial testing is necessary for validation components.** The output guardrail passed its initial fault-injection tests but still contained defects that became visible when it encountered realistic model-generated text. This demonstrated that tests based only on developer-written examples can miss behaviours introduced by real models.

**Guardrails belong on outputs, not only in instructions.** Runtime prompts can tell an LLM not to invent numbers, but the validator provides an independent control over the generated output. This separation is important because a model following an instruction cannot itself guarantee that its output satisfies the application's requirements.

**AI suggestions require verification.** The project demonstrated that both generated implementation changes and generated tests can be incorrect. The appropriate response is not to distrust AI completely, but to place it inside a workflow where suggestions are executed, measured, reviewed, and either accepted or rejected based on evidence.

**The most valuable use of prompting was asking questions that challenged the project.** Asking what could be improved against the actual judging rubric exposed a missing 15-point category before submission. This showed that AI was most useful when used as a development and review partner rather than simply as a code generator.

The overall lesson was that AI can significantly accelerate software development, but **the human developer remains responsible for deciding what to build, validating whether it works, investigating unexpected results, and determining whether the final system is acceptable.**
