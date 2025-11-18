from typing import Any, Callable, Generator, Iterable, List, Optional, Sequence

from presidio_analyzer.nlp_engine import NlpEngine

from metadata.generated.schema.entity.classification.tag import Tag
from metadata.generated.schema.entity.data.table import Column
from metadata.generated.schema.metadataIngestion.workflow import (
    OpenMetadataWorkflowConfig,
)
from metadata.generated.schema.type.tagLabel import (
    LabelType,
    State,
    TagLabel,
    TagSource,
)
from metadata.ingestion.ometa.ometa_api import OpenMetadata
from metadata.pii.algorithms.classifiers import TagClassifier
from metadata.pii.algorithms.presidio_utils import load_nlp_engine
from metadata.pii.algorithms.utils import get_top_classes, normalize_scores
from metadata.pii.base_processor import AutoClassificationProcessor
from metadata.pii.conflict_resolver import ConflictResolver
from metadata.pii.constants import PII
from metadata.pii.models import ScoredTag
from metadata.pii.run_manager import ClassificationRunManager
from metadata.pii.tag_analyzer import TagAnalyzer
from metadata.utils.logger import profiler_logger

logger = profiler_logger()


class TagAnalyzerGenerator:
    def __init__(self, metadata: OpenMetadata, nlp_engine: Optional[NlpEngine] = None):
        self.metadata = metadata
        self.nlp_engine = nlp_engine or load_nlp_engine()
        self._tags: List[Tag] = []

    @property
    def tags(self) -> List[Tag]:
        if not self._tags:
            self._tags = list(
                self.metadata.list_all_entities(
                    entity=Tag,
                    fields=[
                        "name",
                        "recognizers",
                        "fullyQualifiedName",
                        "provider",
                    ],
                )
            )
        return self._tags

    def __call__(self, column: Column) -> Generator[TagAnalyzer, None, None]:
        for tag in self.tags:
            yield TagAnalyzer(tag=tag, column=column, nlp_engine=self.nlp_engine)


class TagProcessor(AutoClassificationProcessor):
    """
    Generic auto-classification processor that supports multiple classifications
    and respects classification-level and tag-level configuration.
    """

    name = "Tag Classification Processor"

    def __init__(
        self,
        config: OpenMetadataWorkflowConfig,
        metadata: OpenMetadata,
        nlp_engine: Optional[NlpEngine] = None,
        classification_filter: Optional[List[str]] = None,
        max_tags_per_column: int = 10,
        tolerance: float = 0.7,
    ) -> None:
        super().__init__(config, metadata)
        self.confidence_threshold = self.source_config.confidence / 100
        self._tolerance = tolerance
        self._nlp_engine = nlp_engine or load_nlp_engine()
        self.classification_filter = classification_filter
        self.max_tags_per_column = max_tags_per_column

        # Initialize new components
        self.run_manager = ClassificationRunManager(metadata)
        self.conflict_resolver = ConflictResolver()

        # Get enabled classifications and their configs
        self.enabled_classifications = self.run_manager.get_enabled_classifications(
            filter_names=classification_filter
        )

        # Get all enabled tags with recognizers from enabled classifications
        self.candidate_tags = self.run_manager.get_enabled_tags(
            classifications=self.enabled_classifications
        )

        logger.info(
            f"TagProcessor initialized with {len(self.enabled_classifications)} "
            f"classifications and {len(self.candidate_tags)} candidate tags"
        )

    def build_tag_label(self, scored_tag: ScoredTag) -> TagLabel:
        """Build a TagLabel from a ScoredTag."""
        tag_label = TagLabel(
            tagFQN=scored_tag.tag.fullyQualifiedName,
            source=TagSource.Classification,
            state=State.Suggested,
            labelType=LabelType.Generated,
        )

        return tag_label

    def create_column_tag_labels(
        self, column: Column, sample_data: Sequence[Any]
    ) -> Sequence[TagLabel]:
        """
        Create tags for the column based on sample data.
        Supports multiple tags from different classifications.
        """
        # Skip if no enabled classifications
        if not self.enabled_classifications:
            logger.debug("No enabled classifications, skipping auto-classification")
            return []

        # Filter candidate tags to exclude already-applied tags
        existing_tag_fqns = {
            tag.tagFQN.root for tag in (column.tags or []) if tag.tagFQN
        }
        tags_to_analyze = [
            tag
            for tag in self.candidate_tags
            if tag.fullyQualifiedName not in existing_tag_fqns
        ]

        if not tags_to_analyze:
            logger.debug(
                f"No new tags to analyze for column {column.name.root} "
                f"(all {len(self.candidate_tags)} candidates already applied)"
            )
            return []

        logger.debug(
            f"Analyzing {len(tags_to_analyze)} tags for column {column.name.root}"
        )

        # Create analyzers for remaining candidate tags
        tag_analyzers = [
            TagAnalyzer(tag=tag, column=column, nlp_engine=self._nlp_engine)
            for tag in tags_to_analyze
        ]

        # Score all tags
        classifier = TagClassifier(tag_analyzers=tag_analyzers)
        scored_tags = classifier.predict_scores(
            sample_data=sample_data,
            column_name=column.fullyQualifiedName,
            column_data_type=column.dataType,
        )

        if not scored_tags:
            logger.debug(
                f"No tags scored above threshold for column {column.name.root}"
            )
            return []

        logger.debug(
            f"Scored {len(scored_tags)} tags for column {column.name.root}, "
            f"top score: {max(scored_tags.values()):.3f}"
        )

        # Apply conflict resolution
        resolved_tags = self.conflict_resolver.resolve_conflicts(
            scored_tags=list(scored_tags.items()),
            enabled_classifications=self.enabled_classifications,
        )

        # Limit total tags per column
        if len(resolved_tags) > self.max_tags_per_column:
            logger.warning(
                f"Column {column.name.root} has {len(resolved_tags)} tags, "
                f"limiting to {self.max_tags_per_column}"
            )
            resolved_tags = sorted(resolved_tags, key=lambda t: t.score, reverse=True)[
                : self.max_tags_per_column
            ]

        logger.info(
            f"Applied {len(resolved_tags)} tags to column {column.name.root}: "
            f"{[t.tag.fullyQualifiedName for t in resolved_tags]}"
        )

        # Build TagLabels
        return [self.build_tag_label(scored_tag) for scored_tag in resolved_tags]
