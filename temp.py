from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
from networks.transformers.roberta_piggyback import PiggybackRobertaForMaskedLM, PiggybackRobertaForSequenceClassification
from config import parseing_posttrain
import torch

config = RobertaConfig()
config.max_position_embeddings = 514
args = parseing_posttrain()
model_trained = PiggybackRobertaForMaskedLM(config, args)
state = torch.load(
    './seq0/640000samples/piggyback/restaurant_unsup_roberta/model.pt')
print(state)
model = PiggybackRobertaForSequenceClassification(config, args, 10)
for n, p in model.roberta.named_parameters():
    print(n)
print(model.roberta.load_state_dict(state, strict=False))
