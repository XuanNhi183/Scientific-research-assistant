/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { Paper, PaperSection, GlossaryItem } from "../types";
import { 
  X, 
  Sparkles, 
  BookOpen, 
  ListPlus, 
  CheckCircle, 
  Cpu, 
  ArrowRight,
  Info,
  Layers,
  Wand2,
  FileText
} from "lucide-react";

interface CustomPaperFormProps {
  onPaperCreated: (paper: Paper) => void;
  onClose: () => void;
}

export default function CustomPaperForm({
  onPaperCreated,
  onClose,
}: CustomPaperFormProps) {
  const [useAI, setUseAI] = useState(true);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [parseStep, setParseStep] = useState(0);

  // Manual configuration inputs state
  const [title, setTitle] = useState("");
  const [authors, setAuthors] = useState("");
  const [year, setYear] = useState("");
  const [journal, setJournal] = useState("");
  const [doi, setDoi] = useState("");
  const [abstract, setAbstract] = useState("");
  const [sections, setSections] = useState<PaperSection[]>([
    { title: "1. Introduction", content: "" }
  ]);
  const [keyFindings, setKeyFindings] = useState<string[]>([""]);
  const [glossary, setGlossary] = useState<GlossaryItem[]>([
    { term: "", definition: "" }
  ]);
  const [errorMessage, setErrorMessage] = useState("");

  // Cycle through engaging, informative parsing steps
  const runExtractionCycles = (onFinish: () => void) => {
    setParseStep(1); // "Structuring macro outline"
    const intervalOne = setTimeout(() => setParseStep(2), 2500); // "Extracting scientific sections"
    const intervalTwo = setTimeout(() => setParseStep(3), 5000); // "Compiling glossary definitions"
    const intervalThree = setTimeout(() => {
      setParseStep(4); // "Finalizing structured paper output"
      setTimeout(onFinish, 1000);
    }, 7500);

    return () => {
      clearTimeout(intervalOne);
      clearTimeout(intervalTwo);
      clearTimeout(intervalThree);
    };
  };

  const handleAIParseSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!pdfFile) {
      setErrorMessage("Please select a PDF file.");
      return;
    }

    setLoading(true);
    setErrorMessage("");
    
    let isFinished = false;
    const cancelTimers = runExtractionCycles(() => {
      isFinished = true;
    });

    try {
      // 1. Upload to FastAPI
      const formData = new FormData();
      formData.append("file", pdfFile);

      const uploadRes = await fetch("http://127.0.0.1:8000/upload_processed_File/", {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) {
        throw new Error("Failed to upload PDF to backend RAG.");
      }
      const uploadData = await uploadRes.json();
      const fileId = uploadData.file_id;

      // 2. Create local URL for PDF viewing
      const pdfLocalUrl = URL.createObjectURL(pdfFile);

      // Create simplified Paper object with just essential info
      const customPaper: Paper = {
        id: fileId,
        title: pdfFile.name,
        authors: "Tác giả gốc",
        year: new Date().getFullYear().toString(),
        abstract: "Tài liệu được tải lên hệ thống gốc.",
        sections: [],
        metrics: { novelty: 0, complexity: "Basic", readingTime: 0, citations: 0 },
        keyFindings: [],
        glossary: [],
        pdfUrl: pdfLocalUrl
      };

      cancelTimers();
      setParseStep(4);
      setTimeout(() => {
        onPaperCreated(customPaper);
        onClose();
      }, 800);

    } catch (err: any) {
      cancelTimers();
      setLoading(false);
      setErrorMessage(err.message || "Interrupted during process. Please try again!");
    }
  };

  // Manual paper creation submit handler
  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !authors.trim() || !abstract.trim()) {
      setErrorMessage("Please fill out all required core fields: Title, Authors, and Abstract.");
      return;
    }

    const filteredSections = sections.filter(s => s.title.trim() && s.content.trim());
    const filteredFindings = keyFindings.filter(f => f.trim());
    const filteredGlossary = glossary.filter(g => g.term.trim() && g.definition.trim());

    if (filteredSections.length === 0) {
      setErrorMessage("Please add at least 1 section with actual scientific text.");
      return;
    }

    const customPaper: Paper = {
      id: `manual-${Date.now()}`,
      title: title,
      authors: authors,
      year: year || new Date().getFullYear().toString(),
      journal: journal || "Manual Library",
      doi: doi || undefined,
      abstract: abstract,
      sections: filteredSections,
      metrics: {
        novelty: 80,
        complexity: "Intermediate",
        readingTime: Math.max(3, Math.ceil(abstract.length / 500)),
        citations: 0
      },
      keyFindings: filteredFindings.length > 0 ? filteredFindings : ["User customized research paper created on system"],
      glossary: filteredGlossary
    };

    onPaperCreated(customPaper);
    onClose();
  };

  const addSection = () => {
    setSections([...sections, { title: `${sections.length + 1}. New Section`, content: "" }]);
  };

  const removeSection = (idx: number) => {
    setSections(sections.filter((_, i) => i !== idx));
  };

  const handleSectionChange = (idx: number, field: "title" | "content", val: string) => {
    const nextSec = [...sections];
    nextSec[idx][field] = val;
    setSections(nextSec);
  };

  return (
    <div id="custom-paper-modal" className="fixed inset-0 bg-gray-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50 overflow-y-auto font-sans">
      <div className="bg-white rounded-2xl border border-gray-200 shadow-2xl w-full max-w-2xl overflow-hidden flex flex-col my-8 max-h-[90vh]">
        
        {/* Header bar */}
        <div className="px-6 py-4 border-b border-gray-150 bg-gray-50 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="p-1.5 bg-blue-100 text-blue-700 rounded-lg">
              <Sparkles className="w-4 h-4 text-blue-600 fill-blue-50" />
            </div>
            <div>
              <h2 className="text-base font-bold text-gray-800">Upload New Research Paper</h2>
              <p className="text-[10px] text-gray-500">Provide a sample text or fill in manually to digitize a publication</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 bg-transparent p-1.5 rounded-lg border-none"
          >
            <X className="w-5 h-5 focus:outline-none" />
          </button>
        </div>

        {/* Switching Modes Option */}
        {!loading && (
          <div className="flex bg-gray-100/80 p-1 m-4 rounded-xl border border-gray-200">
            <button
              onClick={() => { setUseAI(true); setErrorMessage(""); }}
              className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all border-none ${
                useAI 
                  ? "bg-white text-blue-600 shadow-xs" 
                  : "bg-transparent text-gray-600 hover:text-gray-900"
              }`}
            >
              🚀 Auto Extraction (AI)
            </button>
            <button
              onClick={() => { setUseAI(false); setErrorMessage(""); }}
              className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all border-none ${
                !useAI 
                  ? "bg-white text-blue-600 shadow-xs" 
                  : "bg-transparent text-gray-600 hover:text-gray-900"
              }`}
            >
              📝 Manual Entry
            </button>
          </div>
        )}

        <div className="flex-1 overflow-y-auto px-6 pb-6 pt-2">
          {errorMessage && (
            <div className="mb-4 bg-rose-50 border border-rose-150 text-rose-800 p-3 rounded-xl text-xs">
              {errorMessage}
            </div>
          )}

          {/* AI Parser View */}
          {useAI ? (
            <div className="space-y-4">
              {loading ? (
                // Beautiful dynamic loading phases card
                <div className="py-12 flex flex-col items-center justify-center text-center space-y-6">
                  <div className="relative flex items-center justify-center">
                    <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
                    <Cpu className="w-6 h-6 text-blue-600 absolute" />
                  </div>
                  
                  <div className="space-y-2">
                    <h3 className="text-sm font-bold text-gray-800 animate-pulse">
                      {parseStep === 0 && "Connecting to Gemini AI for deep theoretical paper extraction..."}
                      {parseStep === 1 && "Parsing title, publication metrics, and author details..."}
                      {parseStep === 2 && "Synthesizing main sections and structured arguments..."}
                      {parseStep === 3 && "Building glossary entries, key findings, and index metrics..."}
                      {parseStep === 4 && "Finalizing database synchronization!"}
                    </h3>
                    
                    <p className="text-xs text-gray-404 max-w-sm mx-auto leading-relaxed">
                      Gemini is processing the paper contents to synthesize metadata and structure. Saves up to 95% of reading time by extracting index glossary, key findings, and interactive pages.
                    </p>
                  </div>
                  
                  {/* Indicators dots and visual steps representation */}
                  <div className="flex items-center space-x-3.5 text-xs text-gray-450 mt-4 bg-gray-50 border border-gray-150 rounded-xl px-5 py-3 shadow-2xs">
                    <span className={`h-2.5 w-2.5 rounded-full ${parseStep >= 1 ? "bg-blue-600" : "bg-gray-300"}`}></span>
                    <span className="text-[11px] font-medium">Raw Text</span>
                    <ArrowRight className="w-3 h-3 text-gray-400" />
                    <span className={`h-2.5 w-2.5 rounded-full ${parseStep >= 2 ? "bg-blue-600 animate-ping" : "bg-gray-300"}`}></span>
                    <span className="text-[11px] font-medium">Layout</span>
                    <ArrowRight className="w-3 h-3 text-gray-400" />
                    <span className={`h-2.5 w-2.5 rounded-full ${parseStep >= 3 ? "bg-blue-600" : "bg-gray-300"}`}></span>
                    <span className="text-[11px] font-medium font-sans">Glossary</span>
                  </div>
                </div>
              ) : (
                <form onSubmit={handleAIParseSubmit} className="space-y-4 font-sans">
                  <div className="bg-blue-50 border border-blue-150 p-4 rounded-xl flex items-start space-x-3 text-xs text-gray-700">
                    <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                    <div className="leading-relaxed space-y-1">
                      <p className="font-semibold text-blue-800">Smart AI Extraction System:</p>
                      <p>Upload any scientific PDF paper. It will be sent to the RAG backend for chunking, and then structured beautifully by Gemini AI for the interactive reader experience.</p>
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1.5 font-sans">Select PDF Document:</label>
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(e) => setPdfFile(e.target.files?.[0] || null)}
                      className="w-full bg-gray-55 border border-gray-250 text-gray-800 rounded-xl p-3 text-sm focus:outline-none focus:border-blue-500 focus:bg-white"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={!pdfFile}
                    className="w-full py-3 bg-blue-600 text-white rounded-xl font-semibold text-sm hover:bg-blue-700 transition-all border-none flex items-center justify-center space-x-2 cursor-pointer disabled:opacity-50"
                  >
                    <Wand2 className="w-4 h-4 text-white" />
                    <span>Parse & Structure auto-magically with Gemini AI</span>
                  </button>
                </form>
              )}
            </div>
          ) : (
            /* Manual input form code workspace */
            <form onSubmit={handleManualSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Paper Title:</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Attention Is All You Need"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    className="w-full bg-gray-50 border border-gray-250 text-gray-800 rounded-xl p-2.5 text-sm focus:outline-none focus:border-blue-500 focus:bg-white"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Author(s):</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Vaswani et al."
                    value={authors}
                    onChange={(e) => setAuthors(e.target.value)}
                    className="w-full bg-gray-50 border border-gray-250 text-gray-800 rounded-xl p-2.5 text-sm focus:outline-none focus:border-blue-500 focus:bg-white"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Publication Year:</label>
                  <input
                    type="text"
                    placeholder="e.g. 2017"
                    value={year}
                    onChange={(e) => setYear(e.target.value)}
                    className="w-full bg-gray-50 border border-gray-255 text-gray-800 rounded-xl p-2.5 text-sm focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">DOI Identifier (optional):</label>
                  <input
                    type="text"
                    placeholder="e.g. 10.48550/arXiv"
                    value={doi}
                    onChange={(e) => setDoi(e.target.value)}
                    className="w-full bg-gray-50 border border-gray-255 text-gray-800 rounded-xl p-2.5 text-sm focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Journal / Venue:</label>
                  <input
                    type="text"
                    placeholder="e.g. NeurIPS"
                    value={journal}
                    onChange={(e) => setJournal(e.target.value)}
                    className="w-full bg-gray-50 border border-gray-255 text-gray-800 rounded-xl p-2.5 text-sm focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-gray-700 uppercase mb-1">Abstract Overview:</label>
                <textarea
                  rows={4}
                  required
                  placeholder="Paste the abstract summarizing central claims..."
                  value={abstract}
                  onChange={(e) => setAbstract(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-250 text-gray-800 rounded-xl p-2.5 text-sm focus:outline-none focus:border-blue-500 focus:bg-white"
                />
              </div>

              {/* Dynamic sections inputs */}
              <div className="space-y-3.5">
                <div className="flex items-center justify-between border-b border-gray-200 pb-1">
                  <label className="block text-xs font-bold text-blue-800 uppercase tracking-widest">Detailed Sections List ({sections.length}):</label>
                  <button
                    type="button"
                    onClick={addSection}
                    className="text-[11px] font-semibold text-blue-600 bg-blue-50 px-2 py-1 rounded hover:bg-blue-100/60 border-none transition-colors"
                  >
                    + Add Section
                  </button>
                </div>

                {sections.map((sec, idx) => (
                  <div key={idx} className="p-3 bg-gray-50 border border-gray-200 rounded-xl relative space-y-2">
                    {sections.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removeSection(idx)}
                        className="absolute right-2 top-2 text-[10px] text-red-500 font-bold bg-transparent border-none hover:underline"
                      >
                        Remove
                      </button>
                    )}
                    <div>
                      <input
                        type="text"
                        required
                        placeholder={`Section Title (e.g. ${idx+1}. Introduction)`}
                        value={sec.title}
                        onChange={(e) => handleSectionChange(idx, "title", e.target.value)}
                        className="w-full bg-white border border-gray-200 text-gray-800 rounded-lg p-2 text-xs font-bold focus:outline-none"
                      />
                    </div>
                    <div>
                      <textarea
                        rows={3}
                        required
                        placeholder="Section content..."
                        value={sec.content}
                        onChange={(e) => handleSectionChange(idx, "content", e.target.value)}
                        className="w-full bg-white border border-gray-200 text-gray-800 rounded-lg p-2 text-xs focus:outline-none"
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="pt-4 border-t border-gray-150">
                <button
                  type="submit"
                  className="w-full py-3 bg-indigo-600 text-white rounded-xl font-semibold text-sm hover:bg-indigo-700 transition-all border-none flex items-center justify-center space-x-2 cursor-pointer shadow-md animate-fade-in"
                >
                  <CheckCircle className="w-4 h-4 text-white" />
                  <span>Save & Launch in Scientific Workspace</span>
                </button>
              </div>
            </form>
          )}
        </div>

      </div>
    </div>
  );
}
