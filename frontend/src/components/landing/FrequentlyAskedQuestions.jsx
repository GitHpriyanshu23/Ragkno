import React from "react";
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "@/components/ui/accordion";

export const defaultFAQs = [
  {
    question: "What data sources can I connect to RAGKNO?",
    answer:
      "You can upload PDF, DOCX, and TXT files, add an HTTP or HTTPS page, and import supported PDF, TXT, and Google Docs content from Google Drive. The extracted text is chunked and stored in the application's Chroma collection for retrieval.",
  },
  {
    question: "How does RAGKNO cite sources and reduce unsupported answers?",
    answer:
      "RAGKNO asks the model to answer from top-ranked document passages and returns the retrieved source excerpts with the response. Citation buttons let you inspect that evidence, but you should still verify important claims against the original document.",
  },
  {
    question: "What is hybrid search and why does it outperform pure vector search?",
    answer:
      "Dense vector embeddings excel at broad conceptual semantics but often miss exact technical terms, product IDs, or financial figures. RAGKNO pairs dense vector retrieval with BM25 sparse lexical search and fuses them using Reciprocal Rank Fusion (RRF) and cross-encoder reranking for maximum precision.",
  },
  {
    question: "Is chat memory and document data isolated per session?",
    answer:
      "Yes. Threads, messages, memory, sources, and retrieval are scoped to the authenticated user. Multi-turn context stays within the selected thread, and you can clear or delete a conversation at any time.",
  },
  {
    question: "Where is data stored and processed?",
    answer:
      "Document vectors and application metadata are stored by the configured Chroma and SQL backends. Answer generation uses the configured model provider, so retrieved context may be sent to that provider; the current application is not an air-gapped deployment by default.",
  },
  {
    question: "How does RAGKNO handle document structure?",
    answer:
      "RAGKNO extracts text from PDF, DOCX, and TXT documents and preserves page metadata where the parser provides it. Scanned PDFs need OCR before upload, and spreadsheet ingestion is not currently supported.",
  },
  {
    question: "How do you benchmark and evaluate retrieval quality?",
    answer:
      "RAGKNO includes a RAGAS evaluation command for Faithfulness, Answer Relevancy, and Context Precision. It can enforce minimum thresholds against a versioned domain test dataset before a release.",
  },
];

export default function FrequentlyAskedQuestions({
  badge = "FAQ",
  title = "Common questions, clear answers.",
  description = "Everything you need to know about retrieval precision, data privacy, and architecture.",
  data = defaultFAQs,
  className = "",
  supportEmail = "",
}) {
  return (
    <section
      id="faq"
      className={`faq-root-section ${className}`.trim()}
      data-nav-theme="light"
      aria-label="Frequently Asked Questions"
    >
      <div className="faq-container">
        <div className="faq-grid">
          
          {/* Left Column: Badge, Headline & Subtitle */}
          <div className="faq-left-col">
            {badge && <div className="faq-badge">{badge}</div>}

            <h2 className="faq-headline">
              {title === "Common questions, clear answers." ? (
                <>
                  Common questions,<br />clear answers.
                </>
              ) : (
                title
              )}
            </h2>

            <p className="faq-desc">
              {description}
              {supportEmail && (
                <>
                  <br />
                  <a
                    href={`mailto:${supportEmail}`}
                    className="faq-email-link"
                  >
                    {supportEmail}
                  </a>
                </>
              )}
            </p>
          </div>

          {/* Right Column: Accordion list */}
          <div className="faq-right-col">
            <Accordion type="single" collapsible defaultValue="item-0">
              {data.map((item, index) => (
                <AccordionItem key={`faq-${index}`} value={`item-${index}`}>
                  <AccordionTrigger>{item.question}</AccordionTrigger>
                  <AccordionContent>{item.answer}</AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </div>

        </div>
      </div>
    </section>
  );
}
