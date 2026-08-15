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
    source = {
        "source": {
            "name": "SynBioGPT Full-text Literature",
            "type": "collection",
        }
    }

    assert MODULE.resolve_citation_title(metadata, source) == "Actual Paper"


def test_visual_evidence_is_merged_into_its_paper_citation():
    source = {
        "source": {"name": "fulltext", "type": "collection"},
        "document": ["Relevant paragraph.", "Figure 2. Pathway overview."],
        "metadata": [
            {"pmcid": "PMC100", "paper_title": "Paper"},
            {
                "pmcid": "PMC100",
                "paper_title": "Paper",
                "image_urls": ["http://assets/figure-2"],
                "asset_ids": ["Figure 2"],
                "asset_type": "figure",
            },
        ],
    }

    citation = MODULE.build_citation_sources([source])[0]

    assert citation["metadata"]["image_urls"] == ["http://assets/figure-2"]
    assert citation["metadata"]["visual_assets"] == [
        {
            "url": "http://assets/figure-2",
            "label": "Figure 2",
            "caption": "Figure 2. Pathway overview.",
            "asset_type": "figure",
        }
    ]


def test_visual_evidence_is_kept_when_asset_is_the_first_paper_chunk():
    citation = MODULE.build_citation_sources(
        [
            {
                "source": {"name": "fulltext", "type": "collection"},
                "document": ["Table 1. Production results."],
                "metadata": [
                    {
                        "pmcid": "PMC200",
                        "paper_title": "Paper",
                        "image_urls": ["http://assets/table-1"],
                        "asset_ids": ["Table 1"],
                        "asset_type": "table",
                    }
                ],
            }
        ]
    )[0]

    assert citation["metadata"]["visual_assets"][0]["url"] == "http://assets/table-1"


def test_bibliographic_metadata_is_merged_across_retrieval_routes():
    citation = MODULE.build_citation_sources(
        [
            {
                "source": {"name": "fulltext", "type": "collection"},
                "document": ["Dense result.", "BM25 result."],
                "metadata": [
                    {"pmcid": "PMC300", "paper_title": "Paper"},
                    {
                        "pmcid": "PMC300",
                        "paper_title": "Paper",
                        "journal": "Biotechnology Progress",
                        "publication_date": "2016-03-01",
                    },
                ],
            }
        ]
    )[0]

    assert citation["journal"] == "Biotechnology Progress"
    assert citation["publication_date"] == "2016-03-01"
    assert citation["metadata"]["journal"] == "Biotechnology Progress"
