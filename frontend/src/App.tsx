/**
 * @license
 * SPDX-License-Identifier: Apache-2.5
 */

import React, { useState, useEffect, useRef } from "react";
import { Paper, ChatMessage, HistoryItem } from "./types";
import PaperReader from "./components/PaperReader";
import ChatAssistant from "./components/ChatAssistant";
// Upload modal component has been removed in favor of direct file explorer
import {
  BookOpen,
  Sparkles,
  Plus,
  Dna,
  FileText,
  HelpCircle,
  Brain,
  Layers,
  ChevronDown,
  ChevronUp,
  Cpu,
  Info,
  Check,
  History,
  Clock,
  Trash2,
  ChevronsRight
} from "lucide-react";

export default function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaperId, setSelectedPaperId] = useState<string>("");
  const [conversations, setConversations] = useState<{ [paperId: string]: ChatMessage[] }>({});
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [passageReference, setPassageReference] = useState<{ text: string; sectionTitle: string } | null>(null);
  
  const [isLoading, setIsLoading] = useState(false);
  const [showPaperSelectorDropdown, setShowPaperSelectorDropdown] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "loading" | "success" | "error">("idle");

  const [leftWidth, setLeftWidth] = useState<number>(75); // starts at 75%
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isLargeScreen, setIsLargeScreen] = useState<boolean>(true);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDirectFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadStatus("loading");
    setErrorMessage(null);

    try {
      // 1. Upload to FastAPI
      const formData = new FormData();
      formData.append("file", file);

      const uploadRes = await fetch("http://127.0.0.1:8000/upload_processed_File/", {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) {
        throw new Error("Failed to upload PDF to backend RAG.");
      }
      const uploadData = await uploadRes.json();
      const fileId = uploadData.file_id;

      // 2. Use backend URL for PDF viewing to ensure it survives reloads
      const pdfLocalUrl = `http://127.0.0.1:8000/document/${fileId}/pdf`;

      // Create simplified Paper object
      const customPaper: Paper = {
        id: fileId,
        title: file.name,
        authors: "Original Author",
        year: new Date().toLocaleDateString("en-US") + " " + new Date().toLocaleTimeString("en-US"),
        abstract: "Document uploaded to system.",
        sections: [],
        metrics: { novelty: 0, complexity: "Basic", readingTime: 0, citations: 0 },
        keyFindings: [],
        glossary: [],
        pdfUrl: pdfLocalUrl
      };

      handlePaperCreated(customPaper);
      setUploadStatus("success");
      setTimeout(() => {
        setShowUploadModal(false);
        setUploadStatus("idle");
      }, 1500);
    } catch (err: any) {
      setErrorMessage(err.message || "Error uploading file.");
      setUploadStatus("error");
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  useEffect(() => {
    const checkScreen = () => {
      setIsLargeScreen(window.innerWidth >= 1024);
    };
    checkScreen();
    window.addEventListener("resize", checkScreen);
    return () => window.removeEventListener("resize", checkScreen);
  }, []);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const totalWidth = window.innerWidth;
      if (totalWidth <= 0) return;

      let newPercent = (e.clientX / totalWidth) * 100;
      // Clamp the size so that neither side collapses completely (between 20% and 80%)
      newPercent = Math.max(20, Math.min(80, newPercent));
      setLeftWidth(newPercent);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 0) return;
      const touch = e.touches[0];
      const totalWidth = window.innerWidth;
      if (totalWidth <= 0) return;

      let newPercent = (touch.clientX / totalWidth) * 100;
      newPercent = Math.max(20, Math.min(80, newPercent));
      setLeftWidth(newPercent);
    };

    const handleTouchEnd = () => {
      setIsDragging(false);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("touchmove", handleTouchMove, { passive: true });
    window.addEventListener("touchend", handleTouchEnd);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleTouchEnd);
    };
  }, [isDragging]);

  // Initialize papers list from localStorage or PRELOADED data
  useEffect(() => {
    // We changed the key to 'scimind_papers_v2' to automatically clear out old cached data
    const saved = localStorage.getItem("scimind_papers_v2");
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as Paper[];
        if (parsed && parsed.length > 0) {
          setPapers(parsed);
          setSelectedPaperId(parsed[0].id);
          return;
        }
      } catch (e) {
        console.error("Lỗi parse papers từ localStorage", e);
      }
    }
    // Fallback if none parsed
    setPapers([]);
    setSelectedPaperId("");
  }, []);

  // Sync papers to localstorage on edits
  useEffect(() => {
    localStorage.setItem("scimind_papers_v2", JSON.stringify(papers));
  }, [papers]);

  // Load chats state
  useEffect(() => {
    const savedChats = localStorage.getItem("scimind_conversations");
    if (savedChats) {
      try {
        setConversations(JSON.parse(savedChats));
      } catch (e) {
        console.error("Lỗi parse chats", e);
      }
    }
  }, []);

  // Load research activity history
  useEffect(() => {
    const savedHistory = localStorage.getItem("scimind_research_history");
    if (savedHistory) {
      try {
        setHistoryItems(JSON.parse(savedHistory));
      } catch (e) {
        console.error("Lỗi parse history", e);
      }
    } else {
      setHistoryItems([]);
    }
  }, []);

  // Save chats to local storage
  const saveConversation = (paperId: string, msgs: ChatMessage[]) => {
    const updated = {
      ...conversations,
      [paperId]: msgs
    };
    setConversations(updated);
    localStorage.setItem("scimind_conversations", JSON.stringify(updated));
  };

  const activePaper = papers.find(p => p.id === selectedPaperId) || papers[0];
  const activeMessages = activePaper ? (conversations[activePaper.id] || []) : [];

  const handleSendMessage = async (text: string) => {
    if (!activePaper || !text.trim() || isLoading) return;

    // Build the user message object
    const userMsg: ChatMessage = {
      id: `usr-${Date.now()}`,
      sender: "user",
      text: text,
      timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
    };

    if (passageReference) {
      userMsg.passageReference = passageReference;
    }

    const nextMessages = [...activeMessages, userMsg];
    saveConversation(activePaper.id, nextMessages);

    // Save this academic question to the Top History tab state list
    const newHistoryItem: HistoryItem = {
      id: `hist-${Date.now()}`,
      paperId: activePaper.id,
      paperTitle: activePaper.title,
      question: text.length > 90 ? text.substring(0, 90) + "..." : text,
      timestamp: `Today, ${new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}`
    };

    const updatedHistory = [newHistoryItem, ...historyItems.filter(h => h.question !== newHistoryItem.question)].slice(0, 8);
    setHistoryItems(updatedHistory);
    localStorage.setItem("scimind_research_history", JSON.stringify(updatedHistory));

    // Hold reference state then clear immediate client UI selection
    const currentPassageRef = passageReference;
    setPassageReference(null);

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/ask_question/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: currentPassageRef
            ? `Hãy giải thích kỹ đoạn văn tôi chọn này: "${currentPassageRef.text}"\n\nCâu hỏi kèm theo: ${text}`
            : text,
          paper_id: activePaper.id,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Lost connection to the RAG backend server.");
      }

      const data = await response.json();

      let finalAnswer = data.answer;
      if (data.sources && data.sources.length > 0) {
        // Lấy danh sách các trang không bị trùng lặp
        const uniquePages = Array.from(new Set(data.sources.map((s: any) => s.page)));
        finalAnswer += `\n\n**Sources:** Trích xuất từ trang ${uniquePages.join(", ")} của tài liệu.`;
      }

      // Build the AI message object
      const aiReply: ChatMessage = {
        id: `ai-${Date.now()}`,
        sender: "ai",
        text: finalAnswer || "Sorry, I could not analyze this request.",
        timestamp: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
      };

      saveConversation(activePaper.id, [...nextMessages, aiReply]);
    } catch (err: any) {
      console.error(err);
      setErrorMessage(err.message || "An unexpected error occurred while generating the answer.");
    } finally {
      setIsLoading(false);
    }
  };

  // Switch to specific paper and immediately ask question when user clicks a past history tag
  const handleSelectHistoryItem = (item: HistoryItem) => {
    // 1. Locate paper
    const foundPaper = papers.find(p => p.id === item.paperId || p.title.toLowerCase().includes(item.paperTitle.toLowerCase()));
    if (foundPaper) {
      setSelectedPaperId(foundPaper.id);
      setPassageReference(null);
    }

    // 2. Trigger quick AI text feedback
    handleSendMessage(item.question);
  };

  const handleClearHistory = () => {
    if (!activePaper) return;
    saveConversation(activePaper.id, []);
  };

  const handleClearTopHistoryList = () => {
    if (confirm("Do you want to clear the top question history list?")) {
      setHistoryItems([]);
      localStorage.removeItem("scimind_research_history");
    }
  };

  const handlePaperCreated = (newPaper: Paper) => {
    const updatedPapers = [newPaper, ...papers];
    setPapers(updatedPapers);
    setSelectedPaperId(newPaper.id);
  };

  const handleSelectPassage = (text: string, sectionTitle: string) => {
    setPassageReference({ text, sectionTitle });
  };

  return (
    <div id="scimind-main-container" className={`h-screen bg-[#fafaf9] flex flex-col font-sans text-gray-800 selection:bg-blue-100 selection:text-blue-900 overflow-hidden relative ${isDragging ? "select-none cursor-col-resize" : ""}`}>
      
      {!activePaper && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-gray-900/60 backdrop-blur-md overflow-hidden">
          <div className="text-center space-y-6 p-8 max-w-lg w-full bg-white rounded-3xl shadow-2xl relative z-10 animate-in fade-in zoom-in duration-300">
            <div className="w-20 h-20 bg-blue-50 rounded-2xl flex items-center justify-center mx-auto shadow-inner border border-white">
              <Sparkles className="w-10 h-10 text-blue-500" />
            </div>
            <div className="space-y-3">
              <h2 className="text-2xl md:text-3xl font-extrabold text-gray-800 tracking-tight">Welcome to ResearchOS</h2>
              <p className="text-sm md:text-base text-gray-500 leading-relaxed max-w-md mx-auto">
                Your intelligent assistant for scientific research. Upload a PDF paper to extract insights, formulas, and deep contextual understanding instantly.
              </p>
            </div>
            <div className="pt-4">
              <button
                onClick={() => {
                  setShowUploadModal(true);
                  setUploadStatus("idle");
                  setErrorMessage(null);
                }}
                className="w-full sm:w-auto mx-auto px-8 py-3.5 bg-blue-500 hover:bg-blue-600 text-white rounded-xl font-bold shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 transition-all flex items-center justify-center space-x-2 border-none cursor-pointer transform hover:-translate-y-0.5"
              >
                <Plus className="w-5 h-5" />
                <span>Upload your first paper</span>
              </button>
            </div>
            
            <div className="pt-6 mt-6 border-t border-gray-100 grid grid-cols-3 gap-4 text-center">
              <div className="flex flex-col items-center space-y-1.5">
                <div className="w-8 h-8 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center"><BookOpen className="w-4 h-4" /></div>
                <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Analyze</span>
              </div>
              <div className="flex flex-col items-center space-y-1.5">
                <div className="w-8 h-8 bg-amber-50 text-amber-500 rounded-full flex items-center justify-center"><Brain className="w-4 h-4" /></div>
                <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Understand</span>
              </div>
              <div className="flex flex-col items-center space-y-1.5">
                <div className="w-8 h-8 bg-rose-50 text-rose-500 rounded-full flex items-center justify-center"><Layers className="w-4 h-4" /></div>
                <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Extract</span>
              </div>
            </div>
          </div>
        </div>
      )}


      {/* Top Professional Header Navigation */}
      <header className="sticky top-0 bg-white/95 backdrop-blur-md border-b border-gray-200 py-3 px-6 z-40 shadow-2xs flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-3 select-none">
          <div className="w-10 h-10 bg-blue-500 rounded-xl flex items-center justify-center text-white shadow-md">
            <Brain className="w-5.5 h-5.5" />
          </div>
          <div>
            <div className="flex items-center space-x-1.5">
              <span className="text-base font-serif font-extrabold text-gray-900 tracking-tight">ResearchOS</span>
            </div>
            <p className="text-[10px] text-gray-500 font-sans tracking-wide">Scientific Theory Analysis & Academic Q&A Assistant</p>
          </div>
        </div>

        {/* Global Controls inside the main Header */}
        <div className="flex items-center space-x-2 md:space-x-3.5">
          {/* Custom elegant dropdown list */}
          <div className="relative">
            <button
              id="btn-paper-dropdown"
              onClick={() => setShowPaperSelectorDropdown(!showPaperSelectorDropdown)}
              className="px-3 py-1.5 md:px-4 md:py-2 bg-gray-50 border border-gray-250/90 rounded-xl hover:bg-gray-100/70 transition-colors text-xs font-bold text-gray-800 flex items-center space-x-2 border-gray-200 max-w-[150px] sm:max-w-xs md:max-w-md lg:max-w-lg truncate cursor-pointer shadow-2xs"
            >
              <BookOpen className="w-3.5 h-3.5 text-blue-500 flex-shrink-0" />
              <span className="truncate">{activePaper ? "Available Papers" : "Loading..."}</span>
              <ChevronDown className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
            </button>

            {showPaperSelectorDropdown && (
              <div
                id="papers-dropdown-menu"
                className="absolute right-0 mt-2 bg-white border border-gray-200 rounded-xl shadow-xl w-72 md:w-96 overflow-hidden z-50 text-xs py-1"
              >
                <div className="px-3 py-2 bg-gray-50/50 border-b border-gray-100 text-gray-400 font-semibold tracking-wider uppercase text-[10px] pr-8">
                  Available Research Papers
                </div>

                <div className="max-h-80 overflow-y-auto">
                  {papers.map((p) => (
                    <div key={p.id} className="relative group w-full flex items-center pr-8">
                      <button
                        onClick={() => {
                          setSelectedPaperId(p.id);
                          setPassageReference(null);
                          setShowPaperSelectorDropdown(false);
                        }}
                        className={`flex-1 w-full px-4 py-2.5 text-left transition-colors flex items-start space-x-2 bg-transparent border-none pr-8 ${p.id === selectedPaperId
                            ? "bg-blue-50/70 text-blue-600 font-bold"
                            : "text-gray-700 hover:bg-gray-50"
                          }`}
                      >
                        <FileText className={`w-3.5 h-3.5 mt-0.5 flex-shrink-0 ${p.id === selectedPaperId ? "text-blue-500" : "text-gray-400"}`} />
                        <div className="truncate">
                          <div className="truncate font-sans text-xs">{p.title || "Untitled Document"}</div>
                          <div className="text-[10px] text-gray-400 font-normal mt-0.5">
                            Uploaded: {p.year || "Unknown"}
                          </div>
                        </div>
                        {p.id === selectedPaperId && (
                          <Check className="w-3.5 h-3.5 text-blue-500 ml-auto flex-shrink-0 self-center" />
                        )}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          const newPapers = papers.filter(paper => paper.id !== p.id);
                          setPapers(newPapers);
                          if (selectedPaperId === p.id) {
                            setSelectedPaperId(newPapers.length > 0 ? newPapers[0].id : "");
                          }
                        }}
                        className="absolute right-2 p-1.5 text-gray-400 hover:text-rose-500 hover:bg-rose-50 rounded-lg opacity-0 group-hover:opacity-100 transition-all border-none bg-transparent"
                        title="Delete paper"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Create database paper custom */}
          <button
            id="btn-upload-paper"
            onClick={() => {
              setShowUploadModal(true);
              setUploadStatus("idle");
              setErrorMessage(null);
            }}
            className="px-3.5 py-1.5 md:px-4 md:py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-xl text-xs font-bold transition-all border-none flex items-center space-x-1.5 justify-center shadow-xs cursor-pointer hover:shadow-md"
          >
            <Plus className="w-4 h-4 text-white" />
            <span className="hidden sm:inline">Upload paper</span>
            <span className="inline sm:hidden">Upload</span>
          </button>
        </div>
      </header>

      {/* Primary Workspace Layout */}
      <main className="flex-1 p-1 lg:pl-1 lg:pr-2 lg:py-2 flex flex-col lg:flex-row items-stretch max-w-none w-full overflow-hidden min-h-0 relative">


        {/* Left Double pane split column: Paper Reader */}
        <div
          id="workspace-reader-panel"
          className="flex flex-col min-h-0 h-full overflow-hidden shrink-0"
              style={{ width: isLargeScreen ? "50%" : "100%" }}
            >
              {activePaper?.pdfUrl ? (
                <div className="flex-1 w-full h-full bg-white rounded-2xl overflow-hidden border border-gray-200 shadow-sm relative">
                  <div className="absolute top-0 left-0 right-0 h-10 bg-gray-100 border-b border-gray-200 flex items-center px-4 font-semibold text-sm text-gray-700 z-10">
                    <FileText className="w-4 h-4 mr-2 text-blue-500" />
                    <span className="truncate">{activePaper?.title}</span>
                  </div>
                  <iframe
                    src={`${activePaper?.pdfUrl}#navpanes=0&zoom=80`}
                    title={activePaper?.title}
                    className="w-full h-full border-0 pt-10"
                  />
                </div>
              ) : activePaper ? (
                <PaperReader
                  paper={activePaper}
                  onSelectPassage={handleSelectPassage}
                  selectedText={passageReference}
                  onClearSelection={() => setPassageReference(null)}
                />
              ) : (
                <div className="flex-1 w-full h-full bg-white rounded-2xl border border-gray-200 flex flex-col items-center justify-center text-gray-400">
                  <FileText className="w-12 h-12 mb-3 text-gray-300" />
                  <p>No paper selected</p>
                </div>
              )}
            </div>

            {/* Small spacing separator */}
            {(!isLargeScreen || !activePaper) && <div className="h-3 lg:w-3 shrink-0" />}

            {/* Right Double pane split column: AI chat Assistant container */}
            <div
              id="workspace-chat-panel"
              className="flex flex-col min-h-0 h-full overflow-hidden shrink-0"
              style={{ width: isLargeScreen ? "50%" : "100%" }}
            >
              <ChatAssistant
                paper={activePaper}
                messages={activeMessages}
                onSendMessage={handleSendMessage}
                isLoading={isLoading}
                onClearHistory={handleClearHistory}
                passageReference={passageReference}
                onClearPassage={() => setPassageReference(null)}
                historyItems={activePaper ? historyItems.filter(h => h.paperId === activePaper.id) : []}
                onSelectHistoryItem={handleSelectHistoryItem}
              />
            </div>
      </main>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-900/60 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-md mx-4 p-6 relative flex flex-col items-center text-center animate-in fade-in zoom-in duration-200">
            {uploadStatus === "idle" && (
              <>
                <button 
                  onClick={() => setShowUploadModal(false)}
                  className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 bg-transparent"
                >
                  <Plus className="w-5 h-5 rotate-45" />
                </button>
                <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center mb-4 text-blue-500">
                  <FileText className="w-8 h-8" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 mb-2">Upload Research Paper</h3>
                <p className="text-sm text-gray-500 mb-6">Select a PDF file from your computer to analyze</p>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleDirectFileUpload} 
                  accept=".pdf" 
                  className="hidden" 
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-4 rounded-xl shadow-md transition-colors border-none cursor-pointer flex items-center justify-center space-x-2"
                >
                  <Plus className="w-5 h-5" />
                  <span>Select PDF file</span>
                </button>
              </>
            )}

            {uploadStatus === "loading" && (
              <div className="py-8 flex flex-col items-center">
                <div className="w-16 h-16 border-4 border-blue-100 border-t-blue-600 rounded-full animate-spin mb-4"></div>
                <h3 className="text-base font-bold text-gray-800 mb-1">Uploading...</h3>
                <p className="text-xs text-gray-500">System is storing and processing the PDF file</p>
              </div>
            )}

            {uploadStatus === "success" && (
              <div className="py-8 flex flex-col items-center">
                <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mb-4">
                  <Check className="w-8 h-8 text-emerald-600" />
                </div>
                <h3 className="text-base font-bold text-gray-800">Upload successful!</h3>
              </div>
            )}

            {uploadStatus === "error" && (
              <div className="py-4 flex flex-col items-center">
                <div className="w-16 h-16 bg-rose-100 rounded-full flex items-center justify-center mb-4">
                  <Info className="w-8 h-8 text-rose-600" />
                </div>
                <h3 className="text-base font-bold text-gray-800 mb-2">Upload Error</h3>
                <p className="text-sm text-rose-600 mb-6">{errorMessage}</p>
                <div className="flex space-x-3 w-full">
                  <button
                    onClick={() => setShowUploadModal(false)}
                    className="flex-1 py-2.5 bg-gray-100 text-gray-700 rounded-xl font-semibold hover:bg-gray-200 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => setUploadStatus("idle")}
                    className="flex-1 py-2.5 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition-colors"
                  >
                    Retry
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
