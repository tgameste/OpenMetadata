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
Test fixtures for auto-classification tests.
"""
import uuid
from typing import Any, List, Sequence
from unittest.mock import Mock

import pytest

from OpenMetadata.ingestion.build.lib.metadata.generated.schema.type.predefinedRecognizer import PredefinedRecognizer
from metadata.generated.schema.entity.classification.classification import (
    AutoClassificationConfig,
    Classification,
    ConflictResolution,
)
from metadata.generated.schema.entity.classification.tag import Tag
from metadata.generated.schema.type import basic, recognizer, predefinedRecognizer
from metadata.generated.schema.type.entityReference import EntityReference
from metadata.generated.schema.type.patternRecognizer import PatternRecognizer
from metadata.generated.schema.type.piiEntity import PIIEntity
from metadata.generated.schema.type.recognizers.patterns import Pattern
from metadata.generated.schema.type.recognizers.regexFlags import RegexFlags
from metadata.pii.models import ClassificationRunConfig, ScoredTag


@pytest.fixture
def pii_classification() -> Classification:
    """PII classification with auto-classification enabled."""
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
def general_classification() -> Classification:
    """General classification with auto-classification enabled."""
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
def disabled_classification() -> Classification:
    """Classification with auto-classification disabled."""
    return Classification(
        id=basic.Uuid(root=uuid.uuid4()),
        name=basic.EntityName(root="Disabled"),
        fullyQualifiedName="Disabled",
        description=basic.Markdown(root="Disabled classification"),
        mutuallyExclusive=False,
        autoClassificationConfig=AutoClassificationConfig(
            enabled=False,
            conflictResolution=ConflictResolution.highest_confidence,
            minimumConfidence=0.6,
            requireExplicitMatch=True,
        ),
    )


@pytest.fixture
def email_tag_pii(pii_classification: Classification) -> Tag:
    """Email tag in PII classification."""
    return Tag(
        id=basic.Uuid(root=uuid.uuid4()),
        name=basic.EntityName(root="Email"),
        fullyQualifiedName="PII.Email",
        description=basic.Markdown(root="Email address"),
        classification=EntityReference(
            id=pii_classification.id,
            type="classification",
            name=pii_classification.name.root,
            description=pii_classification.description.root,
            fullyQualifiedName=getattr(pii_classification.fullyQualifiedName, "root"),
        ),
        autoClassificationEnabled=True,
        autoClassificationPriority=80,
        recognizers=[
            recognizer.Recognizer(
                name="email-pattern",
                recognizerConfig=recognizer.RecognizerConfig(
                    root=PatternRecognizer(
                        type='pattern',
                        patterns=[
                            Pattern(
                                name="email-pattern",
                                regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
                            )
                        ],
                        regexFlags=RegexFlags(),
                        context=[],
                        supportedEntity=PIIEntity.EMAIL_ADDRESS,
                        supportedLanguage="en",
                    )
                ),
                target=recognizer.Target.content,
            )
        ],
    )


@pytest.fixture
def phone_tag_pii(pii_classification: Classification) -> Tag:
    """Phone tag in PII classification."""
    return Tag(
        id=basic.Uuid(root=uuid.uuid4()),
        name=basic.EntityName(root="Phone"),
        fullyQualifiedName="PII.Phone",
        description=basic.Markdown(root="Phone number"),
        classification=EntityReference(
            id=pii_classification.id,
            type="classification",
            name=pii_classification.name.root,
            description=pii_classification.description.root,
            fullyQualifiedName=getattr(pii_classification.fullyQualifiedName, "root"),
        ),
        autoClassificationEnabled=True,
        autoClassificationPriority=80,
        recognizers=[
            recognizer.Recognizer(
                name="phone-pattern",
                recognizerConfig=recognizer.RecognizerConfig(
                    root=PatternRecognizer(
                        type='pattern',
                        patterns=[
                            Pattern(
                                name="phone-pattern",
                                regex=r"\d{3}-\d{3}-\d{4}",
                            ),
                            Pattern(
                                name="phone-pattern",
                                regex=r"\(\d{3}\)\s*\d{3}-\d{4}",
                            ),
                        ],
                        regexFlags=RegexFlags(),
                        context=[],
                        supportedEntity=PIIEntity.PHONE_NUMBER,
                        supportedLanguage="en",
                    )
                ),
                target=recognizer.Target.content,
            )
        ],
    )


@pytest.fixture
def credit_card_tag_general(general_classification: Classification) -> Tag:
    """Credit Card tag in General classification."""
    return Tag(
        id=basic.Uuid(root=uuid.uuid4()),
        name=basic.EntityName(root="Credit Card"),
        fullyQualifiedName="General.CreditCard",
        description=basic.Markdown(root="Credit Card field"),
        classification=EntityReference(
            id=general_classification.id,
            type="classification",
            name=general_classification.name.root,
            description=general_classification.description.root,
            fullyQualifiedName=getattr(general_classification.fullyQualifiedName, "root"),
        ),
        autoClassificationEnabled=True,
        autoClassificationPriority=90,
        recognizers=[
            recognizer.Recognizer(
                name="credit-card",
                recognizerConfig=recognizer.RecognizerConfig(
                    root=predefinedRecognizer.PredefinedRecognizer(
                        type='predefined',
                        name=predefinedRecognizer.Name.CreditCardRecognizer,
                    )
                ),
                target=recognizer.Target.content,
            )
        ],
    )


@pytest.fixture
def disabled_tag(pii_classification: Classification) -> Tag:
    """Tag with auto-classification disabled."""
    return Tag(
        id=basic.Uuid(root=uuid.uuid4()),
        name=basic.EntityName(root="DisabledTag"),
        fullyQualifiedName="PII.DisabledTag",
        description=basic.Markdown(root="Disabled tag"),
        classification=EntityReference(
            id=pii_classification.id,
            type="classification",
            name=pii_classification.name.root,
            description=pii_classification.description.root,
            fullyQualifiedName=getattr(pii_classification.fullyQualifiedName, "root"),
        ),
        autoClassificationEnabled=False,
        autoClassificationPriority=50,
        recognizers=[],
    )


@pytest.fixture
def tag_without_recognizers(pii_classification: Classification) -> Tag:
    """Tag without recognizers configured."""
    return Tag(
        id=basic.Uuid(root=uuid.uuid4()),
        name=basic.EntityName(root="NoRecognizers"),
        fullyQualifiedName="PII.NoRecognizers",
        description=basic.Markdown(root="Tag without recognizers"),
        classification=EntityReference(
            id=pii_classification.id,
            type="classification",
            name=pii_classification.name.root,
            description=pii_classification.description.root,
            fullyQualifiedName=getattr(pii_classification.fullyQualifiedName, "root"),
        ),
        autoClassificationEnabled=True,
        autoClassificationPriority=50,
        recognizers=None,
    )


@pytest.fixture
def sample_email_data() -> Sequence[Any]:
    """Sample data containing email addresses."""
    return [
        "john.doe@example.com",
        "jane.smith@company.org",
        "admin@test.io",
        "user123@domain.net",
    ]


@pytest.fixture
def sample_phone_data() -> Sequence[Any]:
    """Sample data containing phone numbers."""
    return [
        "555-123-4567",
        "(555) 234-5678",
        "555-345-6789",
        "(555) 456-7890",
    ]


@pytest.fixture
def sample_mixed_data() -> Sequence[Any]:
    """Sample data with mixed content."""
    return [
        "john.doe@example.com",
        "555-123-4567",
        "regular text",
        "jane@company.org",
        "(555) 234-5678",
    ]


@pytest.fixture
def sample_low_cardinality_data() -> Sequence[Any]:
    """Sample data with low cardinality (repeated values)."""
    return ["value1", "value1", "value1", "value2", "value2"]


@pytest.fixture
def mock_metadata_client(mocker) -> Mock:
    """Mocked OpenMetadata client."""
    mock_client = mocker.Mock()
    mock_client.list_all_entities = mocker.Mock(return_value=[])
    return mock_client


@pytest.fixture
def pii_run_config(pii_classification: Classification) -> ClassificationRunConfig:
    """ClassificationRunConfig for PII."""
    return ClassificationRunConfig(
        classification=pii_classification,
        enabled=True,
        min_confidence=0.7,
        conflict_resolution=ConflictResolution.highest_confidence,
        require_explicit_match=True,
    )


@pytest.fixture
def general_run_config(
    general_classification: Classification,
) -> ClassificationRunConfig:
    """ClassificationRunConfig for General."""
    return ClassificationRunConfig(
        classification=general_classification,
        enabled=True,
        min_confidence=0.6,
        conflict_resolution=ConflictResolution.highest_confidence,
        require_explicit_match=True,
    )


@pytest.fixture
def scored_email_tag(email_tag_pii: Tag) -> ScoredTag:
    """ScoredTag for email with high confidence."""
    return ScoredTag(
        tag=email_tag_pii,
        score=0.85,
        classification_name="PII",
        priority=80,
        reason="Detected by Email recognizer: content match (score: 0.85)",
    )


@pytest.fixture
def scored_phone_tag(phone_tag_pii: Tag) -> ScoredTag:
    """ScoredTag for phone with medium confidence."""
    return ScoredTag(
        tag=phone_tag_pii,
        score=0.75,
        classification_name="PII",
        priority=75,
        reason="Detected by Phone recognizer: content match (score: 0.75)",
    )


@pytest.fixture
def scored_password_tag(credit_card_tag_general: Tag) -> ScoredTag:
    """ScoredTag for password with high priority."""
    return ScoredTag(
        tag=credit_card_tag_general,
        score=0.90,
        classification_name="General",
        priority=90,
        reason="Detected by Password recognizer: column name match (score: 0.90)",
    )
