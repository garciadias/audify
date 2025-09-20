"Helper test file to ensure podcast_creator.py is imported for coverage measurement."

import audify.podcast_creator


def test_import_podcast_creator():
    """Test to ensure the module is imported for coverage."""
    assert hasattr(audify.podcast_creator, 'PodcastCreator')
    assert hasattr(audify.podcast_creator, 'LLMClient')
