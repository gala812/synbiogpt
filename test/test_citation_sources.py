import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend/open_webui/apps/retrieval/synbio/citations.py"
)
SPEC = importlib.util.spec_from_file_location("synbio_citations", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_collection_name_is_never_used_as_paper_title():
    source = {
        "source": {
            "name": "SynBioGPT Full-text Literature",
            "type": "collection",
        },
        "document": ["evidence"],
        "metadata": [{"pmcid": "PMC123", "paper_title": ""}],
    }

    citation = MODULE.build_citation_sources([source])[0]
    assert citation["title"] == "PMC123"
    assert citation["source"] == "PMC123"


def test_chunks_from_one_paper_share_one_citation():
    source = {
        "source": {
            "name": "SynBioGPT Full-text Literature",
            "type": "collection",
        },
        "document": ["paragraph", "figure"],
        "metadata": [
            {"pmcid": "PMC456", "paper_title": ""},
            {
                "pmcid": "PMC456",
                "paper_title": "Three-Dimensional Printed Cellulose",
            },
        ],
    }

    citations = MODULE.build_citation_sources([source])
    assert len(citations) == 1
    assert citations[0]["title"] == "Three-Dimensional Printed Cellulose"


def test_paper_title_has_priority_over_generic_title():
    metadata = {
        "pmcid": "PMC789",
        "paper_title": "Actual Paper",
        "title": "SynBioGPT Full-text Literature",
    }
    source = {"source": {"name": "SynBioGPT Full-text Literature", "type": "collection"}}

    assert MODULE.resolve_citation_title(metadata, source) == "Actual Paper"
