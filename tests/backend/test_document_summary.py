from types import SimpleNamespace

import pytest

from src.search import RAGSearch


def search_with_documents():
    rag = RAGSearch.__new__(RAGSearch)
    documents = [{'source_id': 'one', 'source': 'Annual_Report.pdf', 'title': 'Annual Report'},
                 {'source_id': 'two', 'source': 'Research.pdf', 'title': 'Research'}]
    calls = []
    def metadata(user_id, source_ids):
        calls.append((user_id, source_ids))
        return [{'user_id': user_id, 'source_id': source_ids[0], 'source': 'Annual_Report.pdf', 'page': page,
                 'text': f'Page {page}: revenue, risks and company operations.', 'chunk_id': str(page)} for page in range(60)]
    rag.vectorstore = SimpleNamespace(get_user_sources=lambda user_id: documents, get_user_metadata=metadata)
    rag._retrieve_for_answer = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Generic summary must not use relevance search'))
    rag._extract_direct_snippets = lambda *args: []
    rag._invoke_with_fallback = lambda *args: SimpleNamespace(content='The report covers revenue and risks [1].')
    rag._stream_with_fallback = lambda *args: iter(['The report covers ', 'revenue and risks [1].'])
    return rag, documents, calls


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('query', ['summarize the pdf', 'give me a brief summary of this document', 'summarize the pdf again'])
def test_single_document_summarizes_without_filename(stream, query):
    rag, documents, calls = search_with_documents()
    del documents[1:]
    if stream:
        sources, tokens = rag.stream_answer_with_sources(query, user_id='alice')
        answer = ''.join(tokens)
    else:
        result = rag.answer_with_sources(query, user_id='alice')
        sources, answer = result['sources'], result['answer']
    assert 'revenue and risks' in answer
    assert len(sources) == 18
    assert sources[0]['page'] == 0 and sources[-1]['page'] == 59
    assert calls == [('alice', ['one'])]


@pytest.mark.parametrize('stream', [False, True])
def test_multiple_documents_ask_which_one_before_model_call(stream):
    rag, _, calls = search_with_documents()
    if stream:
        sources, tokens = rag.stream_answer_with_sources('summarize the pdf', user_id='alice')
        answer = ''.join(tokens)
    else:
        result = rag.answer_with_sources('summarize the pdf', user_id='alice')
        sources, answer = result['sources'], result['answer']
    assert sources == []
    assert 'Which document' in answer
    assert 'Annual Report' in answer and 'Research' in answer
    assert calls == []


@pytest.mark.parametrize('query,selected,expected', [
    ('summarize the pdf', ['two'], 'two'),
    ('Annual_Report.pdf summarize this pdf', None, 'one'),
    ('summarize Annual Report', None, 'one'),
])
def test_named_or_selected_document_resolves_ambiguity(query, selected, expected):
    rag, _, calls = search_with_documents()
    rag.answer_with_sources(query, user_id='alice', source_ids=selected)
    assert calls == [('alice', [expected])]


def test_specific_topic_summary_uses_regular_retrieval():
    rag, _, calls = search_with_documents()
    assert rag._summary_request_context('summarize revenue growth', 'alice', None) is None
    assert not calls


def test_summary_never_uses_a_source_excluded_by_selection():
    rag, _, calls = search_with_documents()
    result = rag._summary_request_context('summarize Annual Report', 'alice', ['two'])
    assert result is None
    assert not calls
