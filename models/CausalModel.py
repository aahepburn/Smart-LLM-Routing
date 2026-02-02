

class CausalLLMClassifier:
    def __init__(
            self,
            config: RouterModelConfig,
            ckpt_local_path: str,
            prompt_format: PromptFormat,
            score_threshold: int,
            prompt_field: str = "messages",
            use_last_turn: bool = False,
            additional_fields: List[str] = list(["label", "pidx"]),
            max_new_tokens: int = 6,
    ):
        
        assert len(config.special_tokens) == config.num_outputs

    def preprocess(self, row):
        
        return data_row


    def __call__(self, row):
        

        return row


    def compute_routing_prob(self, score_logits):
        pass

    def postprocess(self, row):
        return row
    
    def parse_score(self, text):

        return score


    