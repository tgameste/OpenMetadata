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
Unit tests for ConflictResolver.
"""
from unittest.mock import Mock

import pytest

from metadata.generated.schema.entity.classification.classification import (
    ConflictResolution,
)
from metadata.generated.schema.type import basic
from metadata.pii.conflict_resolver import ConflictResolver
from metadata.pii.models import ClassificationRunConfig, ScoredTag


class TestConflictResolver:
    """Tests for ConflictResolver."""

    def test_resolve_conflicts_empty_list(self, pii_run_config):
        """Test resolving conflicts with empty list."""
        resolver = ConflictResolver()
        resolved = resolver.resolve_conflicts(
            scored_tags=[], enabled_classifications=[pii_run_config]
        )

        assert resolved == []

    def test_resolve_conflicts_mutually_exclusive_highest_confidence(
        self, pii_run_config, email_tag_pii, phone_tag_pii
    ):
        """Test mutually exclusive classification with highest_confidence strategy."""
        # Create tags with different scores
        tag1 = ScoredTag(
            tag=email_tag_pii,
            score=0.85,
            classification_name="PII",
            priority=80,
            reason="Email match",
        )
        tag2 = ScoredTag(
            tag=phone_tag_pii,
            score=0.75,
            classification_name="PII",
            priority=75,
            reason="Phone match",
        )

        resolver = ConflictResolver()
        resolved = resolver.resolve_conflicts(
            scored_tags=[(tag1, 0.85), (tag2, 0.75)],
            enabled_classifications=[pii_run_config],
        )

        # Should return only highest confidence tag (email)
        assert len(resolved) == 1
        assert resolved[0].tag.name.root == "Email"
        assert resolved[0].score == 0.85

    def test_resolve_conflicts_highest_priority(
        self, pii_classification, email_tag_pii, phone_tag_pii
    ):
        """Test conflict resolution with highest_priority strategy."""
        # Update config to use highest_priority
        config = ClassificationRunConfig(
            classification=pii_classification,
            enabled=True,
            min_confidence=0.7,
            conflict_resolution=ConflictResolution.highest_priority,
            require_explicit_match=True,
        )

        # Email has lower score but higher priority
        tag1 = ScoredTag(
            tag=email_tag_pii,
            score=0.75,
            classification_name="PII",
            priority=80,
            reason="Email match",
        )
        # Phone has higher score but lower priority
        tag2 = ScoredTag(
            tag=phone_tag_pii,
            score=0.85,
            classification_name="PII",
            priority=75,
            reason="Phone match",
        )

        resolver = ConflictResolver()
        resolved = resolver.resolve_conflicts(
            scored_tags=[(tag1, 0.75), (tag2, 0.85)],
            enabled_classifications=[config],
        )

        # Should return tag with highest priority (email)
        assert len(resolved) == 1
        assert resolved[0].tag.name.root == "Email"
        assert resolved[0].priority == 80

    def test_resolve_conflicts_most_specific(self, pii_classification):
        """Test conflict resolution with most_specific strategy."""
        # Create tags with different hierarchy depths
        general_tag = Mock()
        general_tag.name = basic.EntityName(root="Sensitive")
        general_tag.fullyQualifiedName = "PII.Sensitive"
        general_tag.classification = Mock(name=basic.EntityName(root="PII"))
        general_tag.autoClassificationPriority = 50

        specific_tag = Mock()
        specific_tag.name = basic.EntityName(root="Email")
        specific_tag.fullyQualifiedName = "PII.Sensitive.Email"
        specific_tag.classification = Mock(name=basic.EntityName(root="PII"))
        specific_tag.autoClassificationPriority = 50

        config = ClassificationRunConfig(
            classification=pii_classification,
            enabled=True,
            min_confidence=0.7,
            conflict_resolution=ConflictResolution.most_specific,
            require_explicit_match=True,
        )

        tag1 = ScoredTag(
            tag=general_tag,
            score=0.85,
            classification_name="PII",
            priority=50,
            reason="General match",
        )
        tag2 = ScoredTag(
            tag=specific_tag,
            score=0.80,
            classification_name="PII",
            priority=50,
            reason="Specific match",
        )

        resolver = ConflictResolver()
        resolved = resolver.resolve_conflicts(
            scored_tags=[(tag1, 0.85), (tag2, 0.80)],
            enabled_classifications=[config],
        )

        # Should return more specific tag (more dots in FQN)
        assert len(resolved) == 1
        assert resolved[0].tag.fullyQualifiedName == "PII.Sensitive.Email"

    def test_resolve_conflicts_non_mutually_exclusive(
        self, general_run_config, credit_card_tag_general
    ):
        """Test non-mutually exclusive classification returns all tags."""
        # Create multiple tags above threshold
        tag1 = ScoredTag(
            tag=credit_card_tag_general,
            score=0.85,
            classification_name="General",
            priority=90,
            reason="Password match",
        )

        secret_tag = Mock()
        secret_tag.name = basic.EntityName(root="Secret")
        secret_tag.fullyQualifiedName = "General.Secret"
        secret_tag.classification = Mock(name=basic.EntityName(root="General"))
        secret_tag.autoClassificationPriority = 85

        tag2 = ScoredTag(
            tag=secret_tag,
            score=0.80,
            classification_name="General",
            priority=85,
            reason="Secret match",
        )

        resolver = ConflictResolver()
        resolved = resolver.resolve_conflicts(
            scored_tags=[(tag1, 0.85), (tag2, 0.80)],
            enabled_classifications=[general_run_config],
        )

        # Should return all tags above threshold
        assert len(resolved) == 2
        tag_names = [t.tag.name.root for t in resolved]
        assert "Credit Card" in tag_names
        assert "Secret" in tag_names

    def test_resolve_conflicts_below_threshold(
        self, pii_run_config, email_tag_pii
    ):
        """Test that tags below minimum confidence are filtered out."""
        # Create tag with score below threshold (0.7)
        tag = ScoredTag(
            tag=email_tag_pii,
            score=0.65,
            classification_name="PII",
            priority=80,
            reason="Weak match",
        )

        resolver = ConflictResolver()
        resolved = resolver.resolve_conflicts(
            scored_tags=[(tag, 0.65)],
            enabled_classifications=[pii_run_config],
        )

        # Should return empty list
        assert resolved == []

    def test_resolve_conflicts_multiple_classifications(
        self,
        pii_run_config,
        general_run_config,
        email_tag_pii,
        credit_card_tag_general,
    ):
        """Test resolving conflicts across multiple classifications."""
        tag1 = ScoredTag(
            tag=email_tag_pii,
            score=0.85,
            classification_name="PII",
            priority=80,
            reason="Email match",
        )
        tag2 = ScoredTag(
            tag=credit_card_tag_general,
            score=0.90,
            classification_name="General",
            priority=90,
            reason="Password match",
        )

        resolver = ConflictResolver()
        resolved = resolver.resolve_conflicts(
            scored_tags=[(tag1, 0.85), (tag2, 0.90)],
            enabled_classifications=[pii_run_config, general_run_config],
        )

        # Should return one tag from each classification
        assert len(resolved) == 2
        classification_names = [t.classification_name for t in resolved]
        assert "PII" in classification_names
        assert "General" in classification_names

    def test_select_winner_single_tag(self, email_tag_pii):
        """Test selecting winner with single tag."""
        tag = ScoredTag(
            tag=email_tag_pii,
            score=0.85,
            classification_name="PII",
            priority=80,
            reason="Email match",
        )

        resolver = ConflictResolver()
        winner = resolver._select_winner(
            [tag], ConflictResolution.highest_confidence
        )

        assert winner == tag

    def test_select_winner_empty_list(self):
        """Test selecting winner with empty list raises error."""
        resolver = ConflictResolver()

        with pytest.raises(ValueError, match="Cannot select winner from empty list"):
            resolver._select_winner([], ConflictResolution.highest_confidence)

    def test_select_winner_tie_breaker(self, email_tag_pii, phone_tag_pii):
        """Test tie-breaking when scores are equal."""
        # Create tags with same score but different priorities
        tag1 = ScoredTag(
            tag=email_tag_pii,
            score=0.85,
            classification_name="PII",
            priority=80,
            reason="Email match",
        )
        tag2 = ScoredTag(
            tag=phone_tag_pii,
            score=0.85,
            classification_name="PII",
            priority=75,
            reason="Phone match",
        )

        resolver = ConflictResolver()
        winner = resolver._select_winner(
            [tag1, tag2], ConflictResolution.highest_confidence
        )

        # With highest_confidence, should use priority as tie-breaker
        assert winner.tag.name.root == "Email"
        assert winner.priority == 80
