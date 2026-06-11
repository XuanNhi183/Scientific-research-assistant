/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from "react";
import { Paper, PaperSection } from "../types";
import { 
  FileText, 
  ZoomIn, 
  ZoomOut, 
  RotateCw, 
  Download, 
  Printer, 
  Bookmark, 
  Layout, 
  Sparkles, 
  ChevronLeft, 
  ChevronRight,
  Info,
  Layers,
  Search,
  BookOpen,
  ChevronsRight,
  Copy,
  Check
} from "lucide-react";

interface PaperReaderProps {
  paper: Paper;
  onSelectPassage: (text: string, sectionTitle: string) => void;
  selectedText: { text: string; sectionTitle: string } | null;
  onClearSelection: () => void;
}

export default function PaperReader({
  paper,
  onSelectPassage,
  selectedText,
  onClearSelection,
}: PaperReaderProps) {
  // Mock PDF states
  const [zoom, setZoom] = useState<number>(100);
  const [rotation, setRotation] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [showHighlightTooltip, setShowHighlightTooltip] = useState<boolean>(false);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [tempSelectedText, setTempSelectedText] = useState<{ text: string; sectionTitle: string } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const pagesContainerRef = useRef<HTMLDivElement>(null);

  // Distribute sections across mock pages to simulate a multi-page PDF document
  const preparePages = (paperObj: Paper) => {
    // We mock that each main section represents a page or two in a standard A4 scientific document
    const pagesList: { pageNum: number; title: string; content: string; sectionObj: PaperSection }[] = [];
    
    // Page 1: Abstract & General Info
    pagesList.push({
      pageNum: 1,
      title: "Abstract & Introduction Overview",
      content: paperObj.abstract,
      sectionObj: { title: "Abstract", content: paperObj.abstract }
    });

    // Subsequent pages mapping to actual sections
    paperObj.sections.forEach((sec, idx) => {
      pagesList.push({
        pageNum: idx + 2,
        title: sec.title,
        content: sec.content,
        sectionObj: sec
      });
    });

    return pagesList;
  };

  const pages = preparePages(paper);
  const totalPages = pages.length;

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 10, 150));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 10, 75));
  const handleRotate = () => setRotation(prev => (prev + 90) % 360);

  // Selection detection for PDF context
  const handlePdfTextSelection = (e: React.MouseEvent, sectionTitle: string) => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim().length > 8) {
      const selectedStr = selection.toString().trim();
      setTempSelectedText({
        text: selectedStr,
        sectionTitle: sectionTitle
      });

      // Position high-tech tooltip above the cursor
      setTooltipPos({
        x: Math.min(e.clientX, window.innerWidth - 180),
        y: e.clientY - 45
      });
      setShowHighlightTooltip(true);
    } else {
      setShowHighlightTooltip(false);
    }
  };

  // Attach passage to parent state so right chatbot handles it
  const handleApplyHighlight = () => {
    if (tempSelectedText) {
      onSelectPassage(tempSelectedText.text, tempSelectedText.sectionTitle);
      setShowHighlightTooltip(false);
    }
  };

  // Highlight matches in current mock text for PDF searching capability
  const highlightSearch = (text: string) => {
    if (!searchQuery.trim()) return text;
    const parts = text.split(new RegExp(`(${searchQuery})`, "gi"));
    return (
      <>
        {parts.map((part, i) => 
          part.toLowerCase() === searchQuery.toLowerCase() ? (
            <mark key={i} className="bg-amber-200 text-gray-900 rounded-xs px-0.5">{part}</mark>
          ) : (
            part
          )
        )}
      </>
    );
  };

  // Scroll mock page into view on thumbnail click
  const scrollToPage = (pageNum: number) => {
    setCurrentPage(pageNum);
    const targetPage = document.getElementById(`pdf-page-${pageNum}`);
    if (targetPage) {
      targetPage.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  // Update current page indicator on scrolling
  const handleContainerScroll = () => {
    if (!pagesContainerRef.current) return;
    const parentTop = pagesContainerRef.current.getBoundingClientRect().top;
    
    // Check which page boundary is currently nearest to center viewport
    let closestPage = 1;
    let minDiff = Infinity;

    pages.forEach(p => {
      const el = document.getElementById(`pdf-page-${p.pageNum}`);
      if (el) {
        const rect = el.getBoundingClientRect();
        const diff = Math.abs(rect.top - parentTop);
        if (diff < minDiff) {
          minDiff = diff;
          closestPage = p.pageNum;
        }
      }
    });

    setCurrentPage(closestPage);
  };

  // Simulation download alert
  const handleDownloadMockPdf = () => {
    alert(`Preparing download for the research PDF: \n"${paper.title}.pdf"\n\nThe document layout has been refined and optimized by ResearchOS.`);
  };

  return (
    <div 
      id="pdf-reader-root" 
      className="flex flex-col h-full bg-[#1e1e1e] border border-gray-700 rounded-2xl overflow-hidden shadow-2xl relative select-none"
      ref={containerRef}
    >
      
      {/* 1. PDF Window Toolbar (Simulating Google Chrome / Adobe reader ribbon) */}
      <div className="bg-[#2d2d2d] border-b border-[#3d3d3d] px-4 py-2.5 flex flex-wrap items-center justify-between text-gray-200 text-xs z-20 gap-2">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <span className="p-1 bg-red-600 text-white rounded font-bold font-sans text-[10px] tracking-wide select-none">PDF</span>
            <span className="font-sans font-semibold text-gray-300 truncate max-w-[150px] md:max-w-xs" title={`${paper.title}.pdf`}>
              {paper.id}.pdf
            </span>
          </div>
        </div>

        {/* Search tool in PDF */}
        <div className="hidden md:flex items-center space-x-1.5 bg-[#1a1a1a] border border-[#3e3e3e] px-2.5 py-1 rounded-lg">
          <Search className="w-3.5 h-3.5 text-gray-400" />
          <input
            type="text"
            placeholder="Search paper content..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-transparent border-none text-white text-xs focus:outline-none w-36 font-sans"
          />
          {searchQuery && (
            <button 
              onClick={() => setSearchQuery("")} 
              className="text-[10px] text-gray-400 bg-transparent hover:text-white"
            >
              x
            </button>
          )}
        </div>

        {/* Zoom and Page Navigation buttons */}
        <div className="flex items-center space-x-3">
          {/* Zoom Controls */}
          <div className="flex items-center bg-[#1e1e1e] rounded-lg border border-[#3e3e3e]">
            <button
              onClick={handleZoomOut}
              disabled={zoom <= 75}
              className="p-1.5 hover:bg-[#3d3d3d] text-gray-300 disabled:opacity-30 rounded-l-lg bg-transparent border-none"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="px-2.5 font-mono text-[11px] font-semibold text-gray-200 text-center min-w-10">
              {zoom}%
            </span>
            <button
              onClick={handleZoomIn}
              disabled={zoom >= 150}
              className="p-1.5 hover:bg-[#3d3d3d] text-gray-300 disabled:opacity-30 rounded-r-lg bg-transparent border-none"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Navigator index of total pages */}
          <div className="flex items-center space-x-1.5 text-gray-300 bg-[#1e1e1e] border border-[#3e3e3e] px-2.5 py-1 rounded-lg font-mono">
            <span>Page</span>
            <span className="font-bold text-white">{currentPage}</span>
            <span className="text-gray-500">/</span>
            <span>{totalPages}</span>
          </div>

          <div className="hidden sm:flex items-center space-x-1 border-l border-[#3d3d3d] pl-2.5">
            <button
              onClick={handleRotate}
              className="p-1.5 hover:bg-[#3d3d3d] text-gray-300 rounded-md bg-transparent border-none"
              title="Rotate page 90°"
            >
              <RotateCw className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleDownloadMockPdf}
              className="p-1.5 hover:bg-[#3d3d3d] text-gray-300 rounded-md bg-transparent border-none"
              title="Download Paper PDF"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* 2. Main Workspace Layout: PDF central canvas */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* PDF pages display viewport - With Scroll and zoom capabilities */}
        <div 
          id="pdf-pages-viewport"
          ref={pagesContainerRef}
          onScroll={handleContainerScroll}
          className="flex-1 overflow-y-auto bg-[#1a1a1a] p-2 md:p-3.5 space-y-8 flex flex-col items-center select-text scroll-smooth"
        >
          {/* Quick PDF instructions guide */}
          <div className="w-full max-w-2xl bg-[#2a2a2a] border border-[#3e3e3e] p-3 rounded-xl text-xs text-gray-300 flex items-start space-x-2.5 shrink-0">
            <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-blue-300">Interactive Reader Workspace:</span> Scroll to read the complete research paper. You can <strong>highlight any phrase or section</strong> to prompt contextual Q&A with the ResearchOS Assistant.
            </div>
          </div>

          {pages.map((p) => (
            <section
              key={p.pageNum}
              id={`pdf-page-${p.pageNum}`}
              className="bg-[#fcfcfa] text-gray-900 border border-black/35 rounded shadow-2xl relative select-text transition-all duration-300 ease-out origin-top shrink-0 px-8 py-10 md:px-12 md:py-14"
              style={{
                width: "100%",
                maxWidth: `${Math.max(480, Math.min(1600, 1350 * (zoom / 100)))}px`,
                transform: `rotate(${rotation}deg)`,
                fontFamily: 'Georgia, Cambria, "Times New Roman", Times, serif',
                minHeight: "1000px"
              }}
              onMouseUp={(e) => handlePdfTextSelection(e, p.title)}
            >
              {/* Header simulation of professional publisher */}
              <div className="border-b border-gray-200 pb-2 mb-6 flex justify-between text-[11px] text-gray-400 font-sans tracking-wider select-none">
                <span className="uppercase font-semibold">ResearchOS Academic Research Portal</span>
                <span>Published: {paper.year || "2024"} | DOI: {paper.id}</span>
              </div>

              {/* Title replication on Page 1 */}
              {p.pageNum === 1 && (
                <div className="mb-8 select-text">
                  <h1 className="text-xl md:text-2xl font-bold font-serif text-gray-900 leading-tight mb-3">
                    {paper.title}
                  </h1>
                  <p className="text-xs text-gray-500 font-sans italic leading-relaxed">
                    Authors: {paper.authors}
                  </p>
                  
                  {paper.journal && (
                    <div className="mt-2 text-[10px] text-indigo-700 bg-indigo-50 font-bold px-2 py-0.5 rounded-md inline-block font-sans select-none border border-indigo-100">
                      Journal: {paper.journal}
                    </div>
                  )}
                </div>
              )}

              {/* Page Section title */}
              <h2 className="text-md md:text-lg font-bold text-blue-950 font-serif mb-4 mt-2 border-b border-gray-100 pb-1.5 select-text">
                {p.title}
              </h2>

              {/* Two-Column layouts formatted beautifully resembling actual arXiv papers */}
              <div 
                className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8 text-sm md:text-base leading-relaxed text-gray-800 text-justify select-text whitespace-pre-line"
                style={{
                  fontFamily: 'serif',
                }}
              >
                {/* Left Column simulated split */}
                <div className="select-text">
                  {highlightSearch(p.content.substring(0, Math.ceil(p.content.length / 2)))}
                </div>

                {/* Right Column simulated split */}
                <div className="select-text">
                  {highlightSearch(p.content.substring(Math.ceil(p.content.length / 2)))}
                </div>
              </div>

              {/* Page Footer decoration */}
              <div className="absolute bottom-6 left-8 right-8 border-t border-gray-100 pt-2 flex justify-between text-[10px] text-gray-400 font-sans select-none">
                <span>ResearchOS Reader Workstation</span>
                <span className="font-semibold font-mono">Page {p.pageNum} / {totalPages}</span>
              </div>
            </section>
          ))}
        </div>
      </div>

      {/* Floating Selection Tooltip Popup relative to caret client context */}
      {showHighlightTooltip && tempSelectedText && (
        <div 
          className="fixed z-50 bg-[#1e1e1e] text-white border border-gray-700 font-sans rounded-xl p-2.5 flex items-center space-x-2 shadow-2xl animate-fade-in text-xs max-w-xs md:max-w-md"
          style={{
            left: `${tooltipPos.x}px`,
            top: `${tooltipPos.y}px`
          }}
        >
          <div className="w-5 h-5 bg-gradient-to-tr from-blue-600 to-indigo-600 rounded-md flex items-center justify-center font-bold text-white shrink-0 shadow-xs">
            <Sparkles className="w-3 h-3 text-white fill-white/20" />
          </div>
          <span className="truncate max-w-[120px] text-gray-300 italic font-mono">&ldquo;{tempSelectedText.text}...&rdquo;</span>
          
          <button
            onClick={handleApplyHighlight}
            className="px-2.5 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-[10px] font-semibold border-none transition-all cursor-pointer"
          >
            Ask AI about this
          </button>
          
          <button
            onClick={() => setShowHighlightTooltip(false)}
            className="text-gray-400 hover:text-white p-1 text-[10px] bg-transparent border-none"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
