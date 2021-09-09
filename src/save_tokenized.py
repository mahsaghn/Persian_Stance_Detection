import logging
import hydra

from feature_extractor import PSFeatureExtractor as FeatureExtractor
from config.baseline_h2c import H2CBaselineConfig, FeatureExtractorConf
from hydra.core.config_store import ConfigStore

logging.basicConfig(filename='main.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

cs = ConfigStore.instance()
cs.store(name="config", node=H2CBaselineConfig)
cs.store(group="",name="baseline_h2c", node=H2CBaselineConfig)
cs.store(group="features",name='features',node=FeatureExtractorConf)



@hydra.main(config_path="config", config_name="config")
def main(cfg: H2CBaselineConfig):
    psf_extractor = FeatureExtractor(cfg=cfg.features)
    tokens_claims, tokens_headlines = psf_extractor.tokenize()
    tclaims = ['__'.join(token_claim)+'\n' for token_claim in tokens_claims]
    claims = psf_extractor.claims
    theadlines = ['__'.join(token_head)+'\n' for token_head in tokens_headlines]
    headlines = psf_extractor.headlines
    f = open('claim_tokens.txt','a')
    f.writelines(tclaims)
    f = open('claims.txt', 'a')
    f.writelines(claims)
    f = open('token_headlines.txt', 'a')
    f.writelines(theadlines)
    f = open('headlines.txt', 'a')
    f.writelines(headlines)
main()