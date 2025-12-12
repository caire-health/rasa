# Slimmed down components for caire-health use case
# Only includes components actually used in rasa-server/config.yml
#
# Removed components (not used by caire-health):
# - MitieIntentClassifier, SklearnIntentClassifier, KeywordIntentClassifier, LogisticRegressionClassifier
# - CRFEntityExtractor, DucklingEntityExtractor, MitieEntityExtractor, SpacyEntityExtractor, RegexEntityExtractor
# - ConveRTFeaturizer, MitieFeaturizer, LanguageModelFeaturizer
# - JiebaTokenizer, MitieTokenizer
# - ResponseSelector
# - UnexpecTEDIntentPolicy, AugmentedMemoizationPolicy
# - MitieNLP

from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.utils.spacy_utils import SpacyNLP

from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.rule_policy import RulePolicy

# Components used by caire-health rasa-server
DEFAULT_COMPONENTS = [
    # NLU Pipeline
    SpacyNLP,
    SpacyTokenizer,
    WhitespaceTokenizer,
    SpacyFeaturizer,
    RegexFeaturizer,
    LexicalSyntacticFeaturizer,
    CountVectorsFeaturizer,
    DIETClassifier,
    EntitySynonymMapper,
    FallbackClassifier,
    # Dialogue Policies
    TEDPolicy,
    MemoizationPolicy,
    RulePolicy,
]
