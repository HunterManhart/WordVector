from cltk.corpus.utils.importer import CorpusImporter
from cltk.corpus.greek.tei import onekgreek_tei_xml_to_text_capitains

corpus_importer = CorpusImporter('greek')

# print(corpus_importer.list_corpora)

corpus_importer.import_corpus('greek_text_first1kgreek')

onekgreek_tei_xml_to_text_capitains()