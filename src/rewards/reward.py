# Local: file src/rewards/reward.py provides first-party ALS source context. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
import re  # Local: imports re for this module. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
# Local: imports selected helpers from termcolor. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
from termcolor import colored
# Local: imports selected helpers from prompts.vera_prompts. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
from prompts.vera_prompts import get_vera_prompt
# Local: imports selected helpers from prompts.vera_prompts. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
from prompts.vera_prompts import VERA_ANSWER_SYMBOL
import torch  # Local: imports torch for this module. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.

# Local: defines the RewardModel class interface. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
class RewardModel(object):
    def __init__(  # Local: defines the __init__ function. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            self, 
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            model,
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            tokenizer,
            # Local: sets data_name: str for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            data_name: str = "gsm8k", 
            # Local: sets device: str for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            device: str = "cuda",
            # Local: sets rule_format_string: str for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            rule_format_string: str = None,
        # Local: closes the surrounding literal or call expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        ):
        # Local: starts a multi-line text literal that Python treats as one value. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        """
        Args:
            model: the model to use for reward prediction
            tokenizer: the tokenizer to use for reward prediction
            data_name
            device
            rule_format_string: str, the answer format that the solution should follow
        """

        # Local: sets self.model for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        self.model = model
        # Local: sets self.tokenizer for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        self.tokenizer = tokenizer

        # Local: sets self.type for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        self.type = type

        # Local: sets self.data_name for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        self.data_name = data_name 
        # Local: sets self.device for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        self.device = device
        # Local: sets self.rule_format_string for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        self.rule_format_string = rule_format_string 



    # Local: defines the load_domain_specific_verifiers function. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
    def load_domain_specific_verifiers(self):
        # Local: sets veras for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        veras = [
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            "calculation_check",
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            "answer_correct",
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            "answer_completeness",
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            "understanding_check",
        ]

        # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        return veras
        

    # Local: defines the get_verifications function. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
    def get_verifications(self, question: str, solution: str):
        # Local: starts a multi-line text literal that Python treats as one value. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        '''
        Get verifications from different verifiers.

        Args:
            question: str, question
            solution: str, solution

        Returns:
            verifications: dict, verifier_name -> verifier_approval
        '''
        # Local: sets veras for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        veras = self.load_domain_specific_verifiers()
        # Local: sets verifications for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        verifications = dict()
        # Local: iterates through the current collection. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        for vera_type in veras:
            # Local: sets vera_prompt for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            vera_prompt = get_vera_prompt(vera_type, question, solution)
            # Local: sets message for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            message = [{"role": "user", "content": vera_prompt}]
            # Local: formats chat messages using the model tokenizer template. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            inputs = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
            # Local: asks the model to generate continuation tokens. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            outputs = self.model.generate(**inputs, max_new_tokens=4096)
            # Local: sets response for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Local: sets verifications[vera_type] for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            verifications[vera_type] = self.extract_verifier_approval(response)

        # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        return verifications
    

    # Local: defines the extract_verifier_approval function. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
    def extract_verifier_approval(self, verifier_response):
        # Local: starts a multi-line text literal that Python treats as one value. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        '''
        Extract verifier approval from verifier response.

        Args:
            verifier_response: str, verifier response

        Returns:
            verifier_approval: bool, verifier approval
        '''
        # Local: sets vera_answer_symbol for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        vera_answer_symbol = VERA_ANSWER_SYMBOL.lower()
        # Local: sets pattern for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        pattern = re.compile(
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            r'.*{}(.*)'.format(re.escape(vera_answer_symbol)), 
            # Local: sets flags for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            flags=re.DOTALL | re.IGNORECASE
        )
        # Local: sets match for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        match = pattern.search(verifier_response)
        # Local: sets answer for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        answer = match.group(1).strip() if match else None
        # Local: opens a condition that selects behavior from current state. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        if not answer:
            # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=}, "
                        # Local: executes this statement in the current code path. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                        f"and full verifier_response (length {len(verifier_response)}): "
                        # Local: executes this statement in the current code path. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                        f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
            # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            return False
    
        # Local: sets answer for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        answer = answer.replace("*", "")  # Remove any asterisks (bolding)
        # Local: sets answer for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        answer = answer.strip().lower()

        # Local: opens a condition that selects behavior from current state. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        if "true" in answer:
            # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            return True
        # Local: checks the next mutually exclusive condition. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        elif "false" in answer:
            # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            return False
        # Local: handles the remaining branch after earlier checks fail. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        else:
            # Check if 'true' or 'false' is in the first word
            # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            print(colored(f"NOTICE in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                        # Local: executes this statement in the current code path. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                        f"checking if the FIRST WORK contains 'true' or 'false'...", "magenta"))
            # Local: sets first_word for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            first_word = answer.split()[0]
            # Local: opens a condition that selects behavior from current state. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            if "true" in first_word:
                # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                print(colored(f"\tSuccess. Found 'true' in first_word.lower(): {first_word.lower()}", "magenta"))
                # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                return True
            # Local: checks the next mutually exclusive condition. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            elif "false" in first_word:
                # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                print(colored(f"\tSuccess. Found 'false' in first_word.lower(): {first_word.lower()}", "magenta"))
                # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                return False
            # Local: handles the remaining branch after earlier checks fail. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            else:
                # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                            # Local: executes this statement in the current code path. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                            f"AND first word does not contain 'true' or 'false. Full verifier_response: "
                            # Local: executes this statement in the current code path. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                            f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
                # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                return False
            

    # Local: defines the get_reward function. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
    def get_reward(self, question, solution):
        # Local: starts a multi-line text literal that Python treats as one value. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        '''
        Get reward from question and solution.

        Args:
            question: str, question
            solution: str, solution
        Returns:
            reward: int, reward
        '''
        # Local: sets verifications for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        verifications = self.get_verifications(question, solution)
        # Local: sets reward for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        reward = 0
        # Local: sets reward_list for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        reward_list = self.get_reward_list()
        # Local: sets total for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        total = 0
        # Local: iterates through the current collection. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        for verifier_name, verifier_approval in verifications.items():
            # Local: updates total for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            total += reward_list[verifier_name]
            # Local: opens a condition that selects behavior from current state. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            if verifier_approval:
                # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                print(colored(f"Verifier {verifier_name} approved the solution.", "green"))
            # Local: handles the remaining branch after earlier checks fail. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            else:
                # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                print(colored(f"Verifier {verifier_name} disapproved the solution.", "red"))
                # Local: updates reward for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                reward -= reward_list[verifier_name]

        # Local: opens a condition that selects behavior from current state. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        if self.rule_format_string is not None:
            # Local: sets format_approval for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            format_approval = self.get_rule_format_verify(solution)
            # Local: opens a condition that selects behavior from current state. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            if format_approval:
                # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                print(colored(f"Verifier Rule Format approved the solution.", "green"))
            # Local: handles the remaining branch after earlier checks fail. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            else:
                # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                print(colored(f"Verifier Rule Format disapproved the solution.", "red"))
                # Local: updates reward for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                reward += -2
                
        # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        return reward / total


    # Local: defines the get_rule_format_verify function. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
    def get_rule_format_verify(self, solution):
        # Local: starts a multi-line text literal that Python treats as one value. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        """
        Judge whether the answer follow the format rule.

        Args:
            solution: str
        """
        # Local: sets answer_pattern for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        answer_pattern = self.rule_format_string
        # Local: applies a regular expression to normalize or extract answer text. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        matches = list(re.finditer(answer_pattern, solution, re.DOTALL))
        # Local: opens a condition that selects behavior from current state. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        if len(matches) > 0:
            # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            return True
        # Local: handles the remaining branch after earlier checks fail. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        else:
            # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            return False
        
    
    # Local: defines the get_reward_answer_only function. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
    def get_reward_answer_only(self, question, solution):
        # Local: starts a multi-line text literal that Python treats as one value. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        '''
        Get reward based only on answer.

        Args:
            question: str, question
            solution: str, solution
        Returns:
            reward: int, reward

        Note that when using this reward function, you should only use the "answer_correct" verifier
        '''
        # Local: sets verifications for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        verifications = self.get_verifications(question, solution)
        # Local: sets reward for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        reward = 0
        # Local: sets reward_list for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        reward_list = self.get_reward_list()
        # Local: sets total for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        total = 0
        # Local: iterates through the current collection. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        for verifier_name, verifier_approval in verifications.items():
            # Local: updates total for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            total += reward_list[verifier_name]
            # Local: opens a condition that selects behavior from current state. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            if verifier_approval:
                # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                print(colored(f"Verifier {verifier_name} approved the solution.", "green"))
            # Local: handles the remaining branch after earlier checks fail. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            else:
                # Local: reports progress or diagnostics to the run log. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                print(colored(f"Verifier {verifier_name} disapproved the solution.", "red"))
                # Local: updates reward for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
                reward -= reward_list[verifier_name]
        # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        return reward / total
        
        
    # Local: defines the get_reward_list function. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
    def get_reward_list(self):
        # Local: starts a multi-line text literal that Python treats as one value. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        '''
        get reward list for different verifiers
        '''
        # Local: sets reward_list for later use in this scope. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        reward_list = {
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            "calculation_check": 2,
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            "answer_correct": 1, 
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            "answer_completeness": 2,
            # Local: adds an item or argument to the surrounding expression. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
            "understanding_check": 1,
        }
        # Local: returns the computed result to the caller. Global: scores generated reasoning with verifier approvals for ALS data collection and LatentSeek.
        return reward_list
