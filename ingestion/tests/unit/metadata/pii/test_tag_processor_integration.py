#  Copyright 2025 Collate
#  Licensed under the Collate Community License, Version 1.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  https://github.com/open-metadata/OpenMetadata/blob/main/ingestion/LICENSE
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Integration tests for TagProcessor with multi-classification support.
Tests scenarios from AUTO_CLASSIFICATION_REFACTOR_SOLUTION.md
"""
import uuid
from typing import Any, Sequence
from unittest.mock import Mock, patch

import pytest

from metadata.generated.schema.entity.classification.classification import (
    AutoClassificationConfig,
    Classification,
    ConflictResolution,
)
from metadata.generated.schema.entity.classification.tag import Tag
from metadata.generated.schema.entity.data.table import Column, DataType
from metadata.generated.schema.metadataIngestion.workflow import (
    OpenMetadataWorkflowConfig,
    SourceConfig,
)
from metadata.generated.schema.type import basic
from metadata.generated.schema.type.entityReference import EntityReference
from metadata.generated.schema.type.patternRecognizer import PatternRecognizer
from metadata.generated.schema.type.piiEntity import PIIEntity
from metadata.generated.schema.type.predefinedRecognizer import PredefinedRecognizer, Name
from metadata.generated.schema.type.recognizer import Recognizer, RecognizerConfig, Target
from metadata.generated.schema.type.recognizers.patterns import Pattern
from metadata.generated.schema.type.recognizers.regexFlags import RegexFlags
from metadata.generated.schema.type.tagLabel import LabelType, State, TagSource
from metadata.pii.tag_processor import TagProcessor


class TestTagProcessorMultiClassification:
    """
    Integration tests for multi-classification scenarios.
    Based on examples from AUTO_CLASSIFICATION_REFACTOR_SOLUTION.md
    """

    @pytest.fixture
    def workflow_config(self):
        """Mock workflow configuration."""
        config = Mock(spec=OpenMetadataWorkflowConfig)
        config.source = Mock(spec=SourceConfig)
        config.source.sourceConfig = Mock()
        config.source.sourceConfig.config = Mock()
        config.source.sourceConfig.config.confidence = 70  # 70% confidence threshold
        return config

    @pytest.fixture
    def pii_classification_mutually_exclusive(self):
        """
        PII Classification (Mutually Exclusive)
        - Only 1 tag can be assigned
        - Uses highest_confidence resolution
        - Minimum confidence: 0.7
        """
        return Classification(
            id=basic.Uuid(root=uuid.uuid4()),
            name=basic.EntityName(root="PII"),
            fullyQualifiedName="PII",
            description=basic.Markdown(root="Personal Identifiable Information"),
            mutuallyExclusive=True,
            autoClassificationConfig=AutoClassificationConfig(
                enabled=True,
                conflictResolution=ConflictResolution.highest_confidence,
                minimumConfidence=0.7,
                requireExplicitMatch=True,
            ),
        )

    @pytest.fixture
    def general_classification_non_exclusive(self):
        """
        General Classification (Non-Mutually Exclusive)
        - Multiple tags can be assigned
        - Minimum confidence: 0.6
        """
        return Classification(
            id=basic.Uuid(root=uuid.uuid4()),
            name=basic.EntityName(root="General"),
            fullyQualifiedName="General",
            description=basic.Markdown(root="General data classifications"),
            mutuallyExclusive=False,
            autoClassificationConfig=AutoClassificationConfig(
                enabled=True,
                conflictResolution=ConflictResolution.highest_confidence,
                minimumConfidence=0.6,
                requireExplicitMatch=True,
            ),
        )

    @pytest.fixture
    def techdetail_classification(self):
        """
        TechDetail Classification (Custom, Non-Mutually Exclusive)
        - Uses highest_priority resolution
        - Minimum confidence: 0.5
        """
        return Classification(
            id=basic.Uuid(root=uuid.uuid4()),
            name=basic.EntityName(root="TechDetail"),
            fullyQualifiedName="TechDetail",
            description=basic.Markdown(root="Technical details"),
            mutuallyExclusive=False,
            autoClassificationConfig=AutoClassificationConfig(
                enabled=True,
                conflictResolution=ConflictResolution.highest_priority,
                minimumConfidence=0.5,
                requireExplicitMatch=True,
            ),
        )

    @pytest.fixture
    def pii_sensitive_tag(self, pii_classification_mutually_exclusive: Classification):
        """PII.Sensitive tag - high priority."""
        return Tag(
            id=basic.Uuid(root=uuid.uuid4()),
            name=basic.EntityName(root="Sensitive"),
            fullyQualifiedName="PII.Sensitive",
            description=basic.Markdown(root="Sensitive data"),
            classification=EntityReference(
                id=pii_classification_mutually_exclusive.id.root,
                type="classification",
                name=pii_classification_mutually_exclusive.name.root,
                description=pii_classification_mutually_exclusive.description.root,
                fullyQualifiedName=getattr(pii_classification_mutually_exclusive.fullyQualifiedName, "root"),
            ),
            autoClassificationEnabled=True,
            autoClassificationPriority=90,
            recognizers=[
                Recognizer(
                    id=basic.Uuid(root=uuid.uuid4()),
                    name="email_recognizer",
                    recognizerConfig=RecognizerConfig(
                        root=PredefinedRecognizer(
                            type="predefined",
                            name=Name.EmailRecognizer
                        )
                    ),
                )
            ],
        )

    @pytest.fixture
    def general_email_tag(self, general_classification_non_exclusive: Classification):
        """General.Email tag."""
        return Tag(
            id=basic.Uuid(root=uuid.uuid4()),
            name=basic.EntityName(root="Email"),
            fullyQualifiedName="General.Email",
            description=basic.Markdown(root="General email classifications"),
            classification=EntityReference(
                id=general_classification_non_exclusive.id.root,
                type="classification",
                name=general_classification_non_exclusive.name.root,
                description=general_classification_non_exclusive.description.root,
                fullyQualifiedName=getattr(general_classification_non_exclusive.fullyQualifiedName, "root"),
            ),
            autoClassificationEnabled=True,
            autoClassificationPriority=95,
            recognizers=[
                Recognizer(
                    id=basic.Uuid(root=uuid.uuid4()),
                    name="email_recognizer",
                    recognizerConfig=RecognizerConfig(
                        root=PredefinedRecognizer(
                            type="predefined",
                            name=Name.EmailRecognizer
                        )
                    ),
                )
            ],
        )

    @pytest.fixture
    def general_password_tag(self, general_classification_non_exclusive: Classification):
        """General.Password tag."""
        return Tag(
            id=basic.Uuid(root=uuid.uuid4()),
            name=basic.EntityName(root="Password"),
            fullyQualifiedName="General.Password",
            description=basic.Markdown(root="General password classifications"),
            classification=EntityReference(
                id=general_classification_non_exclusive.id.root,
                type="classification",
                name=general_classification_non_exclusive.name.root,
                description=general_classification_non_exclusive.description.root,
                fullyQualifiedName=getattr(general_classification_non_exclusive.fullyQualifiedName, "root"),
            ),
            autoClassificationEnabled=True,
            autoClassificationPriority=95,
            recognizers=[
                Recognizer(
                    id=basic.Uuid(root=uuid.uuid4()),
                    name="password_recognizer",
                    recognizerConfig=RecognizerConfig(
                        root=PatternRecognizer(
                            type="pattern",
                            patterns=[
                                Pattern(name="pwd-pattern", regex="^password$")
                            ],
                            regexFlags=RegexFlags(),
                            context=[],
                            supportedEntity=PIIEntity.PERSON,
                            supportedLanguage="en",
                        )
                    ),
                    target=Target.column_name,
                )
            ],
        )

    @pytest.fixture
    def techdetail_secret_tag(self, techdetail_classification: Classification):
        """TechDetail.Secret tag - highest priority."""
        return Tag(
            id=basic.Uuid(root=uuid.uuid4()),
            name=basic.EntityName(root="Secret"),
            fullyQualifiedName="TechDetail.Secret",
            description=basic.Markdown(root="Secret data"),
            classification=EntityReference(
                id=techdetail_classification.id.root,
                type="classification",
                name=techdetail_classification.name.root,
                description=techdetail_classification.description.root,
                fullyQualifiedName=getattr(techdetail_classification.fullyQualifiedName, "root"),
            ),
            autoClassificationEnabled=True,
            autoClassificationPriority=95,
            recognizers=[
                Recognizer(
                    id=basic.Uuid(root=uuid.uuid4()),
                    name="secret_recognizer",
                    recognizerConfig=RecognizerConfig(
                        root=PatternRecognizer(
                            type="pattern",
                            patterns=[
                                Pattern(name="secret-pattern", regex="^secret$")
                            ],
                            regexFlags=RegexFlags(),
                            context=[],
                            supportedEntity=PIIEntity.PERSON,
                            supportedLanguage="en",
                        )
                    ),
                )
            ],
        )

    @pytest.fixture
    def sample_column(self):
        """Sample column with credit card-like data."""
        return Column(
            name="password",
            fullyQualifiedName="database.schema.table.password",
            dataType=DataType.VARCHAR,
            tags=[],
        )

    @pytest.fixture
    def sample_email_password_data(self) -> Sequence[Any]:
        """
        Sample data that could match multiple tags:
        - Contains emails (General.Email)
        - Column name suggests password (General.Password)
        - Contains sensitive data (PII.Sensitive)
        - Could contain secrets (TechDetail.Secret)
        """
        return [
            "user:12dfwef23t1",
            "foo:124dff4y6h44",
            "foobar:9798sfdgs"
        ]

    def test_pii_general_multi_classification(
        self,
        mocker,
        workflow_config,
        pii_classification_mutually_exclusive,
        general_classification_non_exclusive,
        pii_sensitive_tag,
        general_email_tag,
        general_password_tag,
        sample_column,
        sample_email_password_data,
    ):
        """
        Test Example 1 from document: PII + General Multi-Classification

        Expected Result:
        - 1 PII tag (mutually exclusive): PII.Sensitive
        - 2 General tags (non-mutually exclusive): General.Email, General.Password
        """
        # Mock metadata client
        mock_metadata = mocker.Mock()

        # Mock classification fetching
        def list_entities_side_effect(entity, fields=None, params=None):
            if entity == Classification:
                return [
                    pii_classification_mutually_exclusive,
                    general_classification_non_exclusive,
                ]
            elif entity == Tag:
                parent = params.get("parent") if params else None
                if parent == "PII":
                    return [pii_sensitive_tag]
                elif parent == "General":
                    return [general_email_tag, general_password_tag]
            return []

        mock_metadata.list_all_entities.side_effect = list_entities_side_effect

        # Mock NLP engine
        mock_nlp_engine = mocker.Mock()

        # Import ScoredTag before patching
        from metadata.pii.models import ScoredTag

        # Mock TagAnalyzer to avoid recognizer complexity
        with patch("metadata.pii.tag_processor.TagAnalyzer") as mock_analyzer_class:
            # Create mock analyzers that will be instantiated
            mock_analyzers = []
            for tag in [pii_sensitive_tag, general_email_tag, general_password_tag]:
                mock_analyzer = mocker.Mock()
                mock_analyzer.tag = tag
                mock_analyzers.append(mock_analyzer)

            mock_analyzer_class.side_effect = mock_analyzers

            # Mock TagClassifier to return high scores for all tags
            with patch("metadata.pii.tag_processor.TagClassifier") as mock_classifier_class:
                mock_classifier = mocker.Mock()

                # Simulate scores: all tags score above threshold
                mock_scores = {
                    ScoredTag(
                        tag=pii_sensitive_tag,
                        score=0.85,
                        classification_name="PII",
                        priority=90,
                        reason="Detected by Sensitive recognizer: content match",
                    ): 0.85,
                    ScoredTag(
                        tag=general_email_tag,
                        score=0.75,
                        classification_name="General",
                        priority=80,
                        reason="Detected by Email recognizer: content match",
                    ): 0.75,
                    ScoredTag(
                        tag=general_password_tag,
                        score=0.80,
                        classification_name="General",
                        priority=85,
                        reason="Detected by Password recognizer: column name match",
                    ): 0.80,
                }

                mock_classifier.predict_scores.return_value = mock_scores
                mock_classifier_class.return_value = mock_classifier

                # Create TagProcessor
                processor = TagProcessor(
                    config=workflow_config,
                    metadata=mock_metadata,
                    nlp_engine=mock_nlp_engine,
                    classification_filter=None,  # Process all classifications
                    max_tags_per_column=10,
                )

                # Process column
                tag_labels = processor.create_column_tag_labels(
                    column=sample_column, sample_data=sample_email_password_data
                )

                # Verify results
                assert len(tag_labels) == 3, f"Should return 3 tags (1 PII + 2 General), got {len(tag_labels)}: {[l.tagFQN for l in tag_labels]}"

                tag_fqns = [label.tagFQN for label in tag_labels]

                # Should have exactly 1 PII tag (mutually exclusive)
                pii_tags = [fqn.root for fqn in tag_fqns if fqn.root.startswith("PII")]
                assert len(pii_tags) == 1, f"Should have exactly 1 PII tag, got {pii_tags}"
                assert "PII.Sensitive" in pii_tags

                # Should have 2 General tags (non-mutually exclusive)
                general_tags = [fqn.root for fqn in tag_fqns if fqn.root.startswith("General")]
                assert len(general_tags) == 2, f"Should have 2 General tags, got {general_tags}"
                assert "General.Email" in general_tags
                assert "General.Password" in general_tags

                # Verify tag properties
                for label in tag_labels:
                    assert label.source == TagSource.Classification
                    assert label.state == State.Suggested
                    assert label.labelType == LabelType.Generated

    def test_custom_classification_techdetail(
        self,
        mocker,
        workflow_config,
        pii_classification_mutually_exclusive,
        general_classification_non_exclusive,
        techdetail_classification,
        pii_sensitive_tag,
        general_password_tag,
        techdetail_secret_tag,
        sample_column,
        sample_email_password_data,
    ):
        """
        Test Example 2 from document: Custom Classification (TechDetail)

        Expected Result:
        - 1 PII tag: PII.Sensitive
        - 1 General tag: General.Password
        - 1 TechDetail tag: TechDetail.Secret

        Total: 3 tags from 3 different classifications
        """
        # Mock metadata client
        mock_metadata = mocker.Mock()

        # Mock classification fetching - include TechDetail
        def list_entities_side_effect(entity, fields=None, params=None):
            if entity == Classification:
                return [
                    pii_classification_mutually_exclusive,
                    general_classification_non_exclusive,
                    techdetail_classification,
                ]
            elif entity == Tag:
                parent = params.get("parent") if params else None
                if parent == "PII":
                    return [pii_sensitive_tag]
                elif parent == "General":
                    return [general_password_tag]
                elif parent == "TechDetail":
                    return [techdetail_secret_tag]
            return []

        mock_metadata.list_all_entities.side_effect = list_entities_side_effect

        # Mock NLP engine
        mock_nlp_engine = mocker.Mock()

        from metadata.pii.models import ScoredTag

        # Mock TagAnalyzer
        with patch("metadata.pii.tag_processor.TagAnalyzer") as mock_analyzer_class:
            mock_analyzers = []
            for tag in [pii_sensitive_tag, general_password_tag, techdetail_secret_tag]:
                mock_analyzer = mocker.Mock()
                mock_analyzer.tag = tag
                mock_analyzers.append(mock_analyzer)

            mock_analyzer_class.side_effect = mock_analyzers

            # Mock TagClassifier
            with patch("metadata.pii.tag_processor.TagClassifier") as mock_classifier_class:
                mock_classifier = mocker.Mock()

                mock_scores = {
                    ScoredTag(
                        tag=pii_sensitive_tag,
                        score=0.85,
                        classification_name="PII",
                        priority=90,
                        reason="Sensitive data detected",
                    ): 0.85,
                    ScoredTag(
                        tag=general_password_tag,
                        score=0.80,
                        classification_name="General",
                        priority=85,
                        reason="Password field detected",
                    ): 0.80,
                    ScoredTag(
                        tag=techdetail_secret_tag,
                        score=0.75,
                        classification_name="TechDetail",
                        priority=95,  # Highest priority
                        reason="Secret pattern detected",
                    ): 0.75,
                }

                mock_classifier.predict_scores.return_value = mock_scores
                mock_classifier_class.return_value = mock_classifier

                # Create TagProcessor
                processor = TagProcessor(
                    config=workflow_config,
                    metadata=mock_metadata,
                    nlp_engine=mock_nlp_engine,
                    classification_filter=None,
                    max_tags_per_column=10,
                )

                # Process column
                tag_labels = processor.create_column_tag_labels(
                    column=sample_column, sample_data=sample_email_password_data
                )

                # Verify results
                assert len(tag_labels) == 3, f"Should return 3 tags (1 from each classification), got {len(tag_labels)}: {[l.tagFQN for l in tag_labels]}"

                tag_fqns = [label.tagFQN.root for label in tag_labels]

                # Verify each classification contributed 1 tag
                assert "PII.Sensitive" in tag_fqns
                assert "General.Password" in tag_fqns
                assert "TechDetail.Secret" in tag_fqns

    def test_classification_filter(
        self,
        mocker,
        workflow_config,
        pii_classification_mutually_exclusive,
        general_classification_non_exclusive,
        pii_sensitive_tag,
        general_password_tag,
        sample_column,
        sample_email_password_data,
    ):
        """
        Test classification filtering - only process specified classifications.
        """
        mock_metadata = mocker.Mock()

        def list_entities_side_effect(entity, fields=None, params=None):
            if entity == Classification:
                return [
                    pii_classification_mutually_exclusive,
                    general_classification_non_exclusive,
                ]
            elif entity == Tag:
                parent = params.get("parent") if params else None
                if parent == "PII":
                    return [pii_sensitive_tag]
                elif parent == "General":
                    return [general_password_tag]
            return []

        mock_metadata.list_all_entities.side_effect = list_entities_side_effect

        mock_nlp_engine = mocker.Mock()

        from metadata.pii.models import ScoredTag

        # Mock TagAnalyzer
        with patch("metadata.pii.tag_processor.TagAnalyzer") as mock_analyzer_class:
            mock_analyzer = mocker.Mock()
            mock_analyzer.tag = pii_sensitive_tag
            mock_analyzer_class.side_effect = [mock_analyzer]

            with patch("metadata.pii.tag_processor.TagClassifier") as mock_classifier_class:
                mock_classifier = mocker.Mock()

                # Only PII tag will score (General is filtered out)
                mock_scores = {
                    ScoredTag(
                        tag=pii_sensitive_tag,
                        score=0.85,
                        classification_name="PII",
                        priority=90,
                        reason="Sensitive data",
                    ): 0.85,
                }

                mock_classifier.predict_scores.return_value = mock_scores
                mock_classifier_class.return_value = mock_classifier

                # Create TagProcessor with filter - only PII
                processor = TagProcessor(
                    config=workflow_config,
                    metadata=mock_metadata,
                    nlp_engine=mock_nlp_engine,
                    classification_filter=["PII"],  # Only process PII
                    max_tags_per_column=10,
                )

                # Process column
                tag_labels = processor.create_column_tag_labels(
                    column=sample_column, sample_data=sample_email_password_data
                )

                # Should only have PII tag
                assert len(tag_labels) == 1, f"Should only return PII tag, got {len(tag_labels)}: {[l.tagFQN for l in tag_labels]}"
                assert tag_labels[0].tagFQN.root == "PII.Sensitive"

    def test_max_tags_per_column_limit(
        self,
        mocker,
        workflow_config,
        general_classification_non_exclusive,
        sample_column,
        sample_email_password_data,
    ):
        """
        Test that max_tags_per_column limit is enforced.
        """
        mock_metadata = mocker.Mock()

        # Create 5 General tags
        general_tags = []
        for i in range(5):
            tag = Tag(
                id=basic.Uuid(root=uuid.uuid4()),
                name=basic.EntityName(root=f"Tag_{i}"),
                fullyQualifiedName=f"General.Tag_{i}",
                description=basic.Markdown(root=f"Tag {i}'s description"),
                classification=EntityReference(
                    id=general_classification_non_exclusive.id,
                    type="classification",
                    name=general_classification_non_exclusive.name.root,
                    description=general_classification_non_exclusive.description.root,
                    fullyQualifiedName=getattr(general_classification_non_exclusive.fullyQualifiedName, "root"),
                ),
                autoClassificationEnabled=True,
                autoClassificationPriority=80-i,
                recognizers=[
                    Recognizer(
                        id=basic.Uuid(root=uuid.uuid4()),
                        name="email_recognizer",
                        recognizerConfig=RecognizerConfig(
                            root=PredefinedRecognizer(
                                type="predefined",
                                name=Name.EmailRecognizer
                            )
                        ),
                    )
                ],
            )
            general_tags.append(tag)

        def list_entities_side_effect(entity, fields=None, params=None):
            if entity == Classification:
                return [general_classification_non_exclusive]
            elif entity == Tag:
                return general_tags
            return []

        mock_metadata.list_all_entities.side_effect = list_entities_side_effect

        mock_nlp_engine = mocker.Mock()

        from metadata.pii.models import ScoredTag

        # Mock TagAnalyzer
        with patch("metadata.pii.tag_processor.TagAnalyzer") as mock_analyzer_class:
            mock_analyzers = []
            for tag in general_tags:
                mock_analyzer = mocker.Mock()
                mock_analyzer.tag = tag
                mock_analyzers.append(mock_analyzer)

            mock_analyzer_class.side_effect = mock_analyzers

            with patch("metadata.pii.tag_processor.TagClassifier") as mock_classifier_class:
                mock_classifier = mocker.Mock()

                # All 5 tags score above threshold
                mock_scores = {
                    ScoredTag(
                        tag=tag,
                        score=0.70 + i * 0.02,  # Scores: 0.70, 0.72, 0.74, 0.76, 0.78
                        classification_name="General",
                        priority=80 - i,
                        reason=f"Tag{i} detected",
                    ): 0.70 + i * 0.02
                    for i, tag in enumerate(general_tags)
                }

                mock_classifier.predict_scores.return_value = mock_scores
                mock_classifier_class.return_value = mock_classifier

                # Create TagProcessor with limit of 3 tags
                processor = TagProcessor(
                    config=workflow_config,
                    metadata=mock_metadata,
                    nlp_engine=mock_nlp_engine,
                    classification_filter=None,
                    max_tags_per_column=3,  # Limit to 3 tags
                )

                # Process column
                tag_labels = processor.create_column_tag_labels(
                    column=sample_column, sample_data=sample_email_password_data
                )

                # Should only return top 3 tags by score
                assert len(tag_labels) == 3, f"Should limit to 3 tags, got {len(tag_labels)}"

                # Should be the highest scoring tags (Tag4, Tag3, Tag2)
                tag_fqns = [label.tagFQN.root for label in tag_labels]
                assert "General.Tag_4" in tag_fqns  # Highest score: 0.78
                assert "General.Tag_3" in tag_fqns  # Score: 0.76
                assert "General.Tag_2" in tag_fqns  # Score: 0.74

    def test_skip_already_tagged_columns(
        self,
        mocker,
        workflow_config,
        pii_classification_mutually_exclusive,
        pii_sensitive_tag,
        sample_email_password_data,
    ):
        """
        Test that already-applied tags are not re-suggested.
        """
        mock_metadata = mocker.Mock()

        def list_entities_side_effect(entity, fields=None, params=None):
            if entity == Classification:
                return [pii_classification_mutually_exclusive]
            elif entity == Tag:
                return [pii_sensitive_tag]
            return []

        mock_metadata.list_all_entities.side_effect = list_entities_side_effect

        # Column already has PII.Sensitive tag - mock tagFQN properly
        existing_tag_mock = Mock()
        existing_tag_mock.tagFQN = Mock()
        existing_tag_mock.tagFQN.root = "PII.Sensitive"

        column_with_tag = Mock(spec=Column)
        column_with_tag.name = basic.EntityName("user_password")
        column_with_tag.fullyQualifiedName = "database.schema.table.user_password"
        column_with_tag.dataType = DataType.VARCHAR
        column_with_tag.tags = [existing_tag_mock]  # Already has this tag

        mock_nlp_engine = mocker.Mock()

        # Create TagProcessor
        processor = TagProcessor(
            config=workflow_config,
            metadata=mock_metadata,
            nlp_engine=mock_nlp_engine,
            classification_filter=None,
            max_tags_per_column=10,
        )

        # Process column
        tag_labels = processor.create_column_tag_labels(
            column=column_with_tag, sample_data=sample_email_password_data
        )

        # Should return empty list (tag already applied)
        assert len(tag_labels) == 0, f"Should not re-suggest existing tags, got {len(tag_labels)}: {[l.tagFQN for l in tag_labels]}"
