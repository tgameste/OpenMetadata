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
PII processing models
"""
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from metadata.generated.schema.entity.classification.classification import (
    Classification,
    ConflictResolution,
)
from metadata.generated.schema.entity.classification.tag import Tag


class TagType(Enum):
    SENSITIVE = "Sensitive"
    NONSENSITIVE = "NonSensitive"


class TagAndConfidence(BaseModel):
    tag_fqn: str
    confidence: float


@dataclass(frozen=True)
class ScoredTag:
    """
    Result of scoring a tag against sample data.

    Attributes:
        tag: The tag that was scored
        score: Confidence score (0.0-1.0)
        classification_name: Name of the classification this tag belongs to
        priority: Priority for conflict resolution (0-100, higher = higher priority)
        reason: Explanation of why this tag was matched
    """

    tag: Tag
    score: float
    classification_name: str
    priority: int
    reason: str

    def __hash__(self) -> int:
        return hash(self.tag.fullyQualifiedName)


@dataclass(frozen=True)
class ClassificationRunConfig:
    """
    Configuration for a classification during auto-classification run.

    Attributes:
        classification: The classification entity
        enabled: Whether auto-classification is enabled
        min_confidence: Minimum confidence threshold (0.0-1.0)
        conflict_resolution: Strategy for resolving conflicts
        require_explicit_match: Only apply tags when recognizers explicitly match
    """

    classification: Classification
    enabled: bool
    min_confidence: float
    conflict_resolution: ConflictResolution
    require_explicit_match: bool
