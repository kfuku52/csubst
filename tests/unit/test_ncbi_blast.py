import pytest

from csubst import ncbi_blast


XML2_RESULT = """<?xml version="1.0"?>
<BlastXML2 xmlns="http://www.ncbi.nlm.nih.gov">
  <BlastOutput2>
    <report><Report><results><Results><search><Search><hits>
      <Hit><description><HitDescr>
        <id>sp|P12345.7|PROTEIN_ONE</id><accession>P12345.7</accession><title>Protein one</title>
      </HitDescr></description></Hit>
      <Hit><description><HitDescr>
        <id>tr|Q8XYZ1.2|PROTEIN_TWO</id><title>Protein two</title>
      </HitDescr></description></Hit>
      <Hit><description><HitDescr>
        <id>sp|P12345.8|DUPLICATE</id><accession>P12345.8</accession><title>Duplicate</title>
      </HitDescr></description></Hit>
    </hits></Search></search></Results></results></Report></report>
  </BlastOutput2>
</BlastXML2>
"""


class _FakeResponse:
    def __init__(self, text, error=None):
        self.text = text
        self.error = error
        self.closed = False

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, data, headers, timeout):
        self.calls.append({"url": url, "data": dict(data), "headers": dict(headers), "timeout": timeout})
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def test_parse_xml2_hits_uses_accessions_fallbacks_and_deduplicates():
    hits = ncbi_blast.parse_xml2_hits(XML2_RESULT)
    assert [hit.accession for hit in hits] == ["P12345", "Q8XYZ1"]
    assert hits[0].title == "sp|P12345.7|PROTEIN_ONE Protein one"
    assert hits[1].title == "tr|Q8XYZ1.2|PROTEIN_TWO Protein two"


def test_parse_xml2_hits_rejects_malformed_xml():
    with pytest.raises(ValueError, match="malformed XML2"):
        ncbi_blast.parse_xml2_hits("<BlastXML2>")


def test_search_blastp_swissprot_submits_polls_and_closes_responses():
    responses = [
        _FakeResponse("RID = TESTRID123\nRTOE = 12\n"),
        _FakeResponse("Status=WAITING\n"),
        _FakeResponse("Status=READY\n"),
        _FakeResponse(XML2_RESULT),
    ]
    session = _FakeSession(responses)
    sleeps = []
    hits = ncbi_blast.search_blastp_swissprot(
        "AAAA",
        expect=1e-4,
        timeout=17,
        email="maintainer@example.org",
        session=session,
        sleep=sleeps.append,
    )
    assert [hit.accession for hit in hits] == ["P12345", "Q8XYZ1"]
    assert sleeps == [60.0, 60.0, 60.0]
    assert len(session.calls) == 4
    submission = session.calls[0]
    assert submission["data"]["CMD"] == "Put"
    assert submission["data"]["PROGRAM"] == "blastp"
    assert submission["data"]["DATABASE"] == "swissprot"
    assert submission["data"]["HITLIST_SIZE"] == 50
    assert submission["data"]["FORMAT_TYPE"] == "XML2_S"
    assert submission["data"]["email"] == "maintainer@example.org"
    assert submission["data"]["tool"].startswith("csubst-")
    assert session.calls[1]["data"]["RID"] == "TESTRID123"
    assert all(response.closed for response in responses)
    assert session.closed is False


def test_search_blastp_swissprot_uses_ncbi_email_environment(monkeypatch):
    responses = [_FakeResponse("RID = RID123\nRTOE = 60\n"), _FakeResponse(XML2_RESULT)]
    session = _FakeSession(responses)
    monkeypatch.setenv("NCBI_EMAIL", "env@example.org")
    ncbi_blast.search_blastp_swissprot(
        "AAAA",
        session=session,
        sleep=lambda _seconds: None,
    )
    assert session.calls[0]["data"]["email"] == "env@example.org"


def test_search_blastp_swissprot_uses_maintainer_contact_by_default(monkeypatch):
    responses = [_FakeResponse("RID = RID123\nRTOE = 60\n"), _FakeResponse(XML2_RESULT)]
    session = _FakeSession(responses)
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    ncbi_blast.search_blastp_swissprot(
        "AAAA",
        session=session,
        sleep=lambda _seconds: None,
    )
    assert session.calls[0]["data"]["email"] == ncbi_blast.DEFAULT_CONTACT_EMAIL


@pytest.mark.parametrize("status", ["FAILED", "UNKNOWN"])
def test_search_blastp_swissprot_reports_terminal_failure_status(status):
    responses = [_FakeResponse("RID = RID123\nRTOE = 60\n"), _FakeResponse("Status={}\n".format(status))]
    session = _FakeSession(responses)
    with pytest.raises(RuntimeError, match=status):
        ncbi_blast.search_blastp_swissprot(
            "AAAA",
            session=session,
            sleep=lambda _seconds: None,
        )


def test_search_blastp_swissprot_closes_owned_session_on_submission_error(monkeypatch):
    session = _FakeSession([_FakeResponse("submission failed")])
    monkeypatch.setattr(ncbi_blast.requests, "Session", lambda: session)
    with pytest.raises(RuntimeError, match="did not contain an RID"):
        ncbi_blast.search_blastp_swissprot("AAAA", sleep=lambda _seconds: None)
    assert session.closed is True
    assert session.responses == []


def test_search_blastp_swissprot_rejects_too_frequent_polling():
    with pytest.raises(ValueError, match="at least 60 seconds"):
        ncbi_blast.search_blastp_swissprot("AAAA", poll_interval=10)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"timeout": float("nan")}, "request timeout"),
        ({"expect": float("inf")}, "expect value"),
        ({"poll_interval": float("inf")}, "poll_interval"),
    ],
)
def test_search_blastp_swissprot_rejects_non_finite_parameters(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        ncbi_blast.search_blastp_swissprot("AAAA", **kwargs)
