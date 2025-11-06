"Helper test file to ensure audiobook_creator.py is imported for coverage measurement."

import audify.audiobook_creator


def test_import_audiobook_creator():
    """Test to ensure the module is imported for coverage."""
    assert hasattr(audify.audiobook_creator, "PodcastCreator")
    assert hasattr(audify.audiobook_creator, "LLMClient")
