"""Verifier-based reward model for state labeling and LatentSeek.

The ALS workflow uses this wrapper in two places: offline state collection asks
the verifiers whether a generated answer is good enough for the success pool,
and the LatentSeek baseline converts verifier failures into a scalar reward for
per-query hidden-state optimization.
"""

# Regular expressions parse verifier marker text and optional answer-format rules.
import re
# Colored terminal output makes verifier approvals and warnings easier to scan during long runs.
from termcolor import colored
# `get_vera_prompt` builds the verifier-specific prompt text for each check.
from prompts.vera_prompts import get_vera_prompt
# The fixed marker identifies where a verifier's True/False answer begins.
from prompts.vera_prompts import VERA_ANSWER_SYMBOL
# PyTorch is imported for consistency with reward-generation code paths that run model inference.
import torch


# The class wraps an LLM-as-verifier interface behind simple boolean and reward methods.
class RewardModel(object):
    """Run Vera-style verifier prompts and aggregate their decisions.

    Locally, this class generates verifier responses with the same model and
    parses True/False decisions after a marker. Globally, those decisions define
    good/bad hidden-state labels for ALS and scalar rewards for LatentSeek.
    """

    # Constructor stores the shared model, tokenizer, dataset context, and optional format rule.
    def __init__(
            # `self` holds configuration reused across verifier calls.
            self,
            # The model generates verifier responses.
            model,
            # The tokenizer formats verifier prompts and decodes verifier responses.
            tokenizer,
            # Dataset name is stored for compatibility with reward/evaluation contexts.
            data_name: str = "gsm8k",
            # Device places verifier prompt tensors on the same hardware as the model.
            device: str = "cuda",
            # Optional regex enforces answer format, such as boxed output.
            rule_format_string: str = None,
        ):
        """Store verifier dependencies and configuration.

        Locally, no model is loaded here; the already loaded solver model is
        reused. Globally, this keeps reward computation aligned with the exact
        model being evaluated or optimized.
        """

        # Store the model so `get_verifications` can call `generate`.
        self.model = model
        # Store the tokenizer so verifier prompts use the same chat template as solving.
        self.tokenizer = tokenizer

        # Preserve the existing attribute assignment for compatibility, although it is unused.
        self.type = type

        # Store the dataset name for reward-context awareness.
        self.data_name = data_name
        # Store the target device for verifier input tensors.
        self.device = device
        # Store the optional regex used by `get_rule_format_verify`.
        self.rule_format_string = rule_format_string


    # This method defines the verifier ensemble used throughout the repository.
    def load_domain_specific_verifiers(self):
        """Return the verifier names used for mathematical solutions.

        Locally, the names are keys into `get_vera_prompt`. Globally, this list
        defines the dimensions of correctness used to label ALS states and score
        LatentSeek candidates.
        """
        # The list order determines the order in which verifier prompts are run.
        veras = [
            # Calculation checks validate arithmetic/algebraic steps.
            "calculation_check",
            # Answer correctness checks only the final answer.
            "answer_correct",
            # Completeness checks whether a final answer is actually provided.
            "answer_completeness",
            # Understanding checks whether the solution addressed the intended problem.
            "understanding_check",
        ]

        # Returning names keeps prompt construction centralized in `vera_prompts.py`.
        return veras


    # This method runs every verifier and returns parsed approvals.
    def get_verifications(self, question: str, solution: str):
        """Run all verifier prompts for a question/solution pair.

        Locally, each verifier prompt is generated, decoded, and parsed into a
        boolean approval. Globally, the approval dictionary becomes either a
        good/bad label threshold or a weighted reward.
        """
        # Load the verifier names for this domain.
        veras = self.load_domain_specific_verifiers()
        # The dictionary maps each verifier name to its boolean approval.
        verifications = dict()
        # Iterate through verifiers so each dimension of solution quality is checked.
        for vera_type in veras:
            # Build the verifier-specific prompt using the original problem and generated solution.
            vera_prompt = get_vera_prompt(vera_type, question, solution)
            # Wrap the prompt as a single user message for the chat template.
            message = [{"role": "user", "content": vera_prompt}]
            # The tokenizer converts the verifier message into model input tensors on the target device.
            inputs = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
            # The model generates the verifier's analysis and final marker answer.
            outputs = self.model.generate(**inputs, max_new_tokens=4096)
            # Decoding turns the verifier output token ids into text for regex parsing.
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # The parser converts the response into a True/False approval.
            verifications[vera_type] = self.extract_verifier_approval(response)

        # The completed dictionary feeds state labeling or reward aggregation.
        return verifications


    # This parser extracts the boolean verifier decision after the required marker.
    def extract_verifier_approval(self, verifier_response):
        """Parse a verifier response into a boolean approval.

        Locally, the method searches for the configured marker and then accepts
        True/False text after it. Globally, conservative parsing treats malformed
        verifier output as disapproval so bad labels/rewards are not inflated.
        """
        # Lowercasing the marker lets the regex match case-insensitively.
        vera_answer_symbol = VERA_ANSWER_SYMBOL.lower()
        # The regex captures everything after the final marker occurrence.
        pattern = re.compile(
            # Escaping the marker prevents punctuation in the marker from acting as regex syntax.
            r'.*{}(.*)'.format(re.escape(vera_answer_symbol)),
            # DOTALL spans multiline verifier reasoning; IGNORECASE handles marker casing.
            flags=re.DOTALL | re.IGNORECASE
        )
        # Searching the full response finds the marker and captured answer text.
        match = pattern.search(verifier_response)
        # If the marker exists, strip whitespace around the captured answer.
        answer = match.group(1).strip() if match else None
        # Missing answer text is treated as verifier failure.
        if not answer:
            # The warning includes the full response so prompt or parsing failures can be debugged.
            print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=}, "
                        # The response length helps diagnose truncation or unusually long outputs.
                        f"and full verifier_response (length {len(verifier_response)}): "
                        # Separator lines make the raw verifier response readable in logs.
                        f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
            # Malformed verifier output is conservatively counted as disapproval.
            return False

        # Asterisks are removed because models may bold the final True/False token.
        answer = answer.replace("*", "")
        # Lowercasing and stripping normalize variants like ` True`.
        answer = answer.strip().lower()

        # Any occurrence of true after the marker is accepted as approval.
        if "true" in answer:
            # Approval contributes positive verifier count and avoids reward penalty.
            return True
        # Any occurrence of false after the marker is accepted as disapproval.
        elif "false" in answer:
            # Disapproval contributes to bad labels or reward penalties.
            return False
        # If neither token appears globally, try a narrower first-word fallback.
        else:
            # The notice explains the fallback parser path.
            print(colored(f"NOTICE in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                        # The message clarifies that only the first word will be inspected next.
                        f"checking if the FIRST WORK contains 'true' or 'false'...", "magenta"))
            # Splitting isolates the first word after the marker.
            first_word = answer.split()[0]
            # A first word containing true is treated as approval.
            if "true" in first_word:
                # The success message records exactly what matched.
                print(colored(f"\tSuccess. Found 'true' in first_word.lower(): {first_word.lower()}", "magenta"))
                # Return approval after the fallback match.
                return True
            # A first word containing false is treated as disapproval.
            elif "false" in first_word:
                # The success message records exactly what matched.
                print(colored(f"\tSuccess. Found 'false' in first_word.lower(): {first_word.lower()}", "magenta"))
                # Return disapproval after the fallback match.
                return False
            # If no recognizable boolean appears, fail conservatively.
            else:
                # The warning includes raw response text for debugging parser failures.
                print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                            # This line states that even the first-word fallback failed.
                            f"AND first word does not contain 'true' or 'false. Full verifier_response: "
                            # Separator lines make the response readable in logs.
                            f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
                # Malformed verifier output is treated as disapproval.
                return False


    # This method converts verifier decisions into the scalar reward used by LatentSeek.
    def get_reward(self, question, solution):
        """Return a weighted negative-penalty reward for a proposed solution.

        Locally, failed verifiers subtract their weights and passed verifiers
        print approval logs. Globally, this scalar drives LatentSeek's
        per-query hidden-state optimization.
        """
        # Run all verifiers and parse their approvals.
        verifications = self.get_verifications(question, solution)
        # Reward starts at zero and becomes negative for failed checks.
        reward = 0
        # The weight table defines how costly each failed verifier is.
        reward_list = self.get_reward_list()
        # `total` accumulates weights for normalization.
        total = 0
        # Iterate over verifier results to aggregate penalties.
        for verifier_name, verifier_approval in verifications.items():
            # Add this verifier's weight to the normalization denominator.
            total += reward_list[verifier_name]
            # Approved verifiers do not add penalty.
            if verifier_approval:
                # Green output makes approvals visible in optimization logs.
                print(colored(f"Verifier {verifier_name} approved the solution.", "green"))
            # Failed verifiers subtract their configured weight.
            else:
                # Red output makes failures visible in optimization logs.
                print(colored(f"Verifier {verifier_name} disapproved the solution.", "red"))
                # Subtracting the verifier weight lowers reward for this failure mode.
                reward -= reward_list[verifier_name]

        # Optional format rules add an extra penalty for missing required answer syntax.
        if self.rule_format_string is not None:
            # The regex check returns whether the solution contains the required format.
            format_approval = self.get_rule_format_verify(solution)
            # Passing the format rule produces only a log message.
            if format_approval:
                # Green output confirms the rule did not penalize the solution.
                print(colored(f"Verifier Rule Format approved the solution.", "green"))
            # Failing the format rule adds a fixed extra penalty.
            else:
                # Red output highlights schema/format failure.
                print(colored(f"Verifier Rule Format disapproved the solution.", "red"))
                # The fixed penalty discourages candidates that miss boxed or other required answer format.
                reward += -2

        # Normalizing by total verifier weight keeps reward scale stable across verifier weights.
        return reward / total


    # This helper checks optional answer-format constraints.
    def get_rule_format_verify(self, solution):
        """Return whether the solution satisfies the configured regex format.

        Locally, this searches the solution text for `rule_format_string`.
        Globally, the result can add an extra LatentSeek reward penalty when
        prompt format matters, such as boxed final answers.
        """
        # The configured regex pattern describes the required answer shape.
        answer_pattern = self.rule_format_string
        # `finditer` collects every match so any valid occurrence counts.
        matches = list(re.finditer(answer_pattern, solution, re.DOTALL))
        # At least one regex match means the solution satisfies the format rule.
        if len(matches) > 0:
            # Return True so no format penalty is applied.
            return True
        # No regex match means the solution violates the required format.
        else:
            # Return False so the caller can apply a format penalty.
            return False


    # This alternate reward path is kept for experiments focused only on final-answer correctness.
    def get_reward_answer_only(self, question, solution):
        """Return reward while conceptually focusing on answer correctness.

        Locally, this currently aggregates the same verifier dictionary and
        weights as `get_reward`. Globally, it is intended for ablations that use
        only the answer-correct verifier.
        """
        # Run verifiers on the candidate solution.
        verifications = self.get_verifications(question, solution)
        # Reward starts at zero and becomes negative for failed checks.
        reward = 0
        # Fetch the verifier weight table.
        reward_list = self.get_reward_list()
        # Accumulate the denominator used for normalization.
        total = 0
        # Aggregate every verifier decision.
        for verifier_name, verifier_approval in verifications.items():
            # Add this verifier's weight to the denominator.
            total += reward_list[verifier_name]
            # Approved verifiers do not penalize the answer.
            if verifier_approval:
                # Log approval in green.
                print(colored(f"Verifier {verifier_name} approved the solution.", "green"))
            # Failed verifiers subtract their weights.
            else:
                # Log disapproval in red.
                print(colored(f"Verifier {verifier_name} disapproved the solution.", "red"))
                # Subtract the configured failure penalty.
                reward -= reward_list[verifier_name]
        # Return a normalized reward compatible with `get_reward`.
        return reward / total


    # This method defines the penalty weights for each verifier type.
    def get_reward_list(self):
        """Return verifier penalty weights used for reward aggregation.

        Locally, the dictionary weights calculation and completeness higher than
        answer correctness and understanding. Globally, these weights shape the
        reward landscape for LatentSeek and the meaning of verifier thresholds.
        """
        # The dictionary maps verifier names to positive weights that become negative penalties on failure.
        reward_list = {
            # Calculation mistakes receive weight 2.
            "calculation_check": 2,
            # Final-answer mistakes receive weight 1.
            "answer_correct": 1,
            # Incomplete answers receive weight 2.
            "answer_completeness": 2,
            # Misunderstanding the problem receives weight 1.
            "understanding_check": 1,
        }
        # Returning the dictionary lets reward methods share one weighting scheme.
        return reward_list
