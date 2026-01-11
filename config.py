from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json


class TokenizerAlgorithm(Enum):
    """enum for tokenization algorithms"""
    BPE = "bpe"
    WORDPIECE = "wordpiece"
    UNIGRAM = "unigram"
    TOKENMONSTER = "tokenmonster"
    SRE = "sre"
    SRE_HYBRID = "sre_hybrid"
    SRE_REGULARIZED = "sre_regularized"
    TOKENMONSTER_SRE_HYBRID = "tokenmonster_sre_hybrid"


class VocabularyStrategy(Enum):
    """enum for vocab strategies"""
    UNIFIED = "unified"
    #not used
    PARTITIONED = "partitioned"


class Language(Enum):
    """languages of talmud as enum"""
    BIBLICAL_HEBREW = "BH"
    MISHNAIC_HEBREW = "MH"
    JEWISH_BABYLONIAN_ARAMAIC = "JBA"


@dataclass
class TokenizerConfig:
    """base config"""
    algorithm: TokenizerAlgorithm
    vocabulary_strategy: VocabularyStrategy
    use_bpe_dropout: bool
    vocab_size: int = 32000
    dropout_rate: float = 0.1
    
    #see final paper
    min_frequency: int = 2
    alpha_renyi: float = 2.5
    
    #language-informed parameters
    aramaic_upsampling_factor: float = 2.0

    
    #SRE-specific stuff
    morphological_analyzer: str = "dicta"
    
    def to_dict(self) -> Dict:
        """converts the config to dictionary."""
        return {
            "algorithm": self.algorithm.value,
            "vocabulary_strategy": self.vocabulary_strategy.value,
            "use_bpe_dropout": self.use_bpe_dropout,
            "vocab_size": self.vocab_size,
            "dropout_rate": self.dropout_rate,
            "min_frequency": self.min_frequency,
            "alpha_renyi": self.alpha_renyi,
            "aramaic_upsampling_factor": self.aramaic_upsampling_factor,
            "morphological_analyzer": self.morphological_analyzer,
        }
    
    def get_name(self) -> str:
        """make a unique name for the config"""
        dropout_str = "dropout" if self.use_bpe_dropout else "nodropout"
        return f"{self.algorithm.value}_{self.vocabulary_strategy.value}_{dropout_str}"


@dataclass
class ExperimentConfig:
    """config for the entire experiment!"""
    #you can use whichever corpus you want
    corpus_path: str = "sample_talmud_corpus.txt"
    output_dir: str = "./results"

    #this can be file or sefaria_full
    corpus_source: str = "file"
    #none is all; can specify
    tractates: Optional[List[str]] = None  
    max_words_per_segment: int = 40
    min_words_per_segment: int = 3
    cache_talmud: bool = True
    
    #configs to test
    algorithms: List[TokenizerAlgorithm] = field(default_factory=lambda: list(TokenizerAlgorithm))
    vocabulary_strategies: List[VocabularyStrategy] = field(default_factory=lambda: list(VocabularyStrategy))
    use_dropout_options: List[bool] = field(default_factory=lambda: [True, False])
    
    #other params
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    vocab_size: int = 32000
    
    #eval params
    mlm_epochs: int = 10
    mlm_batch_size: int = 32
    mlm_mask_prob: float = 0.15
    
    #vis params
    plot_style: str = "seaborn-v0_8"
    figure_dpi: int = 300
    
    #parallelization params; none is autodetect
    num_workers: Optional[int] = None
    use_parallel: bool = True
    
    def generate_all_configs(self) -> List[TokenizerConfig]:
        """Generates all of the configs"""
        configs = []
        
        #if using unannotated full Talmud then filter by vocab strat
        vocab_strategies = self.vocabulary_strategies
        if self.corpus_source == "sefaria_full":
            #unified for unanotated
            vocab_strategies = [vs for vs in vocab_strategies if vs == VocabularyStrategy.UNIFIED]
            if not vocab_strategies:
                print("Warning: No valid vocabulary strategies for sefaria_full corpus were found; you must use unified!!")
        
        for algo in self.algorithms:
            for vocab_strategy in vocab_strategies:
                for use_dropout in self.use_dropout_options:
                    config = TokenizerConfig(
                        algorithm=algo,
                        vocabulary_strategy=vocab_strategy,
                        use_bpe_dropout=use_dropout,
                        vocab_size=self.vocab_size
                    )
                    configs.append(config)
        return configs
    
    def save(self, path: str):
        """config --> json"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'corpus_path': self.corpus_path,
                'output_dir': self.output_dir,
                'corpus_source': self.corpus_source,
                'tractates': self.tractates,
                'max_words_per_segment': self.max_words_per_segment,
                'min_words_per_segment': self.min_words_per_segment,
                'cache_talmud': self.cache_talmud,
                'train_split': self.train_split,
                'val_split': self.val_split,
                'test_split': self.test_split,
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """from JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)