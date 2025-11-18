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
Unit tests for ClassificationRunManager.
"""
import pytest

from metadata.generated.schema.entity.classification.classification import (
    Classification,
    ConflictResolution,
)
from metadata.generated.schema.entity.classification.tag import Tag
from metadata.pii.run_manager import ClassificationRunManager


class TestClassificationRunManager:
    """Tests for ClassificationRunManager."""

    def test_get_enabled_classifications(
        self,
        mocker,
        pii_classification,
        general_classification,
        disabled_classification,
    ):
        """Test fetching enabled classifications."""
        mock_metadata = mocker.Mock()
        mock_metadata.list_all_entities.return_value = [
            pii_classification,
            general_classification,
            disabled_classification,
        ]

        manager = ClassificationRunManager(mock_metadata)
        enabled = manager.get_enabled_classifications()

        # Should return only enabled classifications (PII and General)
        assert len(enabled) == 2
        classification_names = [c.classification.name.root for c in enabled]
        assert "PII" in classification_names
        assert "General" in classification_names
        assert "Disabled" not in classification_names

        # Verify configs are populated correctly
        pii_config = next(c for c in enabled if c.classification.name.root == "PII")
        assert pii_config.min_confidence == 0.7
        assert pii_config.conflict_resolution == ConflictResolution.highest_confidence
        assert pii_config.enabled is True

    def test_get_enabled_classifications_with_filter(
        self, mocker, pii_classification, general_classification
    ):
        """Test fetching enabled classifications with name filter."""
        mock_metadata = mocker.Mock()
        mock_metadata.list_all_entities.return_value = [
            pii_classification,
            general_classification,
        ]

        manager = ClassificationRunManager(mock_metadata)
        enabled = manager.get_enabled_classifications(filter_names=["PII"])

        # Should return only PII
        assert len(enabled) == 1
        assert enabled[0].classification.name.root == "PII"

    def test_get_enabled_classifications_caching(
        self, mocker, pii_classification
    ):
        """Test that classifications are cached."""
        mock_metadata = mocker.Mock()
        mock_metadata.list_all_entities.return_value = [pii_classification]

        manager = ClassificationRunManager(mock_metadata)

        # First call
        enabled1 = manager.get_enabled_classifications()
        # Second call
        enabled2 = manager.get_enabled_classifications()

        # Should only call API once
        assert mock_metadata.list_all_entities.call_count == 1
        assert enabled1 == enabled2

    def test_get_enabled_tags(
        self,
        mocker,
        pii_run_config,
        email_tag_pii,
        phone_tag_pii,
        disabled_tag,
        tag_without_recognizers,
    ):
        """Test fetching enabled tags with recognizers."""
        mock_metadata = mocker.Mock()
        mock_metadata.list_all_entities.return_value = [
            email_tag_pii,
            phone_tag_pii,
            disabled_tag,
            tag_without_recognizers,
        ]

        manager = ClassificationRunManager(mock_metadata)
        tags = manager.get_enabled_tags(classifications=[pii_run_config])

        # Should return only enabled tags with recognizers
        assert len(tags) == 2
        tag_names = [t.name.root for t in tags]
        assert "Email" in tag_names
        assert "Phone" in tag_names
        assert "DisabledTag" not in tag_names
        assert "NoRecognizers" not in tag_names

    def test_get_enabled_tags_multiple_classifications(
        self,
        mocker,
        pii_run_config,
        general_run_config,
        email_tag_pii,
        credit_card_tag_general,
    ):
        """Test fetching tags from multiple classifications."""
        mock_metadata = mocker.Mock()

        def list_entities_side_effect(entity, fields, params):
            if params.get("parent") == "PII":
                return [email_tag_pii]
            elif params.get("parent") == "General":
                return [credit_card_tag_general]
            return []

        mock_metadata.list_all_entities.side_effect = list_entities_side_effect

        manager = ClassificationRunManager(mock_metadata)
        tags = manager.get_enabled_tags(
            classifications=[pii_run_config, general_run_config]
        )

        # Should return tags from both classifications
        assert len(tags) == 2
        tag_fqns = {t.fullyQualifiedName for t in tags}
        assert "PII.Email" in tag_fqns
        assert "General.CreditCard" in tag_fqns

    def test_get_enabled_tags_caching(self, mocker, pii_run_config, email_tag_pii):
        """Test that tags are cached."""
        mock_metadata = mocker.Mock()
        mock_metadata.list_all_entities.return_value = [email_tag_pii]

        manager = ClassificationRunManager(mock_metadata)

        # First call
        tags1 = manager.get_enabled_tags(classifications=[pii_run_config])
        # Second call
        tags2 = manager.get_enabled_tags(classifications=[pii_run_config])

        # Should only call API once
        assert mock_metadata.list_all_entities.call_count == 1
        assert tags1 == tags2

    def test_clear_cache(self, mocker, pii_classification, email_tag_pii):
        """Test clearing the cache."""
        mock_metadata = mocker.Mock()
        mock_metadata.list_all_entities.return_value = [pii_classification]

        manager = ClassificationRunManager(mock_metadata)

        # First call
        manager.get_enabled_classifications()
        assert mock_metadata.list_all_entities.call_count == 1

        # Clear cache
        manager.clear_cache()

        # Second call should hit API again
        manager.get_enabled_classifications()
        assert mock_metadata.list_all_entities.call_count == 2

    def test_get_enabled_classifications_api_error(self, mocker):
        """Test handling of API errors."""
        mock_metadata = mocker.Mock()
        mock_metadata.list_all_entities.side_effect = Exception("API Error")

        manager = ClassificationRunManager(mock_metadata)
        enabled = manager.get_enabled_classifications()

        # Should return empty list on error
        assert enabled == []

    def test_get_enabled_tags_api_error(self, mocker, pii_run_config):
        """Test handling of API errors when fetching tags."""
        mock_metadata = mocker.Mock()
        mock_metadata.list_all_entities.side_effect = Exception("API Error")

        manager = ClassificationRunManager(mock_metadata)
        tags = manager.get_enabled_tags(classifications=[pii_run_config])

        # Should return empty list on error
        assert tags == []
