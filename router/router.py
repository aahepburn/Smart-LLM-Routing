import abc


class Router(abc.ABC):

    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak
        
    def __str__(self):
        return NAME_TO_CLS[self.__class__]
    

class CausalLLMRouter(Router):
    def __init__(
            self,
            checkpoint_path,
            score_threshold=4,
            special_tokens="",
            model_type='causal',
            model_id = "",
            flash_attention_2=False,
    ):
        model_config = RouterModelConfig()