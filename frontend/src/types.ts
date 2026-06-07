/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface PaperSection {
  title: string;
  content: string;
}

export interface PaperMetrics {
  novelty: number; // 0 to 100
  complexity: "Basic" | "Intermediate" | "Advanced";
  readingTime: number; // minutes
  citations: number;
}

export interface GlossaryItem {
  term: string;
  definition: string;
}

export interface Paper {
  id: string;
  title: string;
  authors: string;
  year: string;
  journal?: string;
  doi?: string;
  abstract: string;
  sections: PaperSection[];
  metrics: PaperMetrics;
  keyFindings: string[];
  glossary: GlossaryItem[];
  pdfUrl?: string;
}

export interface ChatMessage {
  id: string;
  sender: "user" | "ai";
  text: string;
  timestamp: string;
  passageReference?: {
    text: string;
    sectionTitle: string;
  };
}

export interface HistoryItem {
  id: string;
  paperId: string;
  paperTitle: string;
  question: string;
  answerSummary?: string;
  timestamp: string;
}

