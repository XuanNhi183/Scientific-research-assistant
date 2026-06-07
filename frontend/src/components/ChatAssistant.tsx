/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useRef, useEffect } from "react";
import { ChatMessage, Paper, HistoryItem } from "../types";
import {
  Send,
  Sparkles,
  Cpu,
  User,
  Trash2,
  HelpCircle,
  XCircle,
  CornerDownRight,
  RefreshCw,
  Maximize2,
  Clock
} from "lucide-react";

interface ChatAssistantProps {
  paper: Paper | null;
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  isLoading: boolean;
  onClearHistory: () => void;
  passageReference: { text: string; sectionTitle: string } | null;
  onClearPassage: () => void;
  historyItems?: HistoryItem[];
  onSelectHistoryItem?: (item: HistoryItem) => void;
}

export default function ChatAssistant({
  paper,
  messages,
  onSendMessage,
  isLoading,
  onClearHistory,
  passageReference,
  onClearPassage,
  historyItems = [],
  onSelectHistoryItem,
}: ChatAssistantProps) {
  const [inputText, setInputText] = React.useState("");
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Handle form submission
  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;
    onSendMessage(inputText);
    setInputText("");
  };

  // Preset prompts tailored per paper
  const getPresetPrompts = (paperId: string): string[] => {
    switch (paperId) {
      case "attention-is-all-you-need":
        return [
          "How does the Self-Attention mechanism work, and why is it parallelizable?",
          "Why is the dot-product divided by the square root of d_k (Scaled Dot-Product)?",
          "What function is used for Positional Encoding and what is its role?"
        ];
      case "resnet-image-recognition":
        return [
          "What is the degradation problem, and how is it different from overfitting?",
          "How does Residual learning F(x) + x resolve the degradation problem?",
          "How does Identity Mapping operate without adding extra learning parameters?"
        ];
      case "generative-adversarial-nets":
        return [
          "How is the minimax game of GANs mathematically formulated?",
          "Why doesn't the Generator need Markov Chain sampling or integral approximation?",
          "What are the adversarial roles of the objective functions for D(x) and G(z)?"
        ];
      default:
        return [
          "Summarize the main scientific contributions of this research paper.",
          "What is the key model design or methodology proposed in this document?",
          "What limitations or future directions of research are discussed?"
        ];
    }
  };

  const presetPrompts = getPresetPrompts(paper?.id);

  // Elite formatter translating basic markdown, bullet points, headers, inline codes and block equations beautifully
  const renderFormattedMessage = (text: string) => {
    const lines = text.split("\n");
    return lines.map((line, idx) => {
      let trimmed = line.trim();

      // Check code blocks
      if (trimmed.startsWith("```")) {
        return null; // Skip markdown wrappers visually
      }

      // Check header H3
      if (trimmed.startsWith("###")) {
        return (
          <h4 key={idx} className="text-sm font-bold text-gray-800 mt-3 mb-1.5 font-sans flex items-center">
            <span className="w-1.5 h-3.5 bg-blue-500 rounded mr-2 inline-block"></span>
            {trimmed.replace(/^###\s*/, "")}
          </h4>
        );
      }

      // Check header H2
      if (trimmed.startsWith("##")) {
        return (
          <h3 key={idx} className="text-base font-serif font-bold text-blue-900 mt-4 mb-2">
            {trimmed.replace(/^##\s*/, "")}
          </h3>
        );
      }

      // Check bullet items
      if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
        const cleanContent = trimmed.replace(/^[-*]\s*/, "");
        return (
          <ul key={idx} className="list-disc pl-5 my-1 text-gray-700 space-y-1">
            <li className="text-sm md:text-base leading-relaxed">{parseInlineStyles(cleanContent)}</li>
          </ul>
        );
      }

      // Check numbered items
      const numberedMatch = trimmed.match(/^(\d+)\.\s(.*)/);
      if (numberedMatch) {
        return (
          <ol key={idx} className="list-decimal pl-5 my-1 text-gray-700 space-y-1">
            <li className="text-sm md:text-base leading-relaxed">{parseInlineStyles(numberedMatch[2])}</li>
          </ol>
        );
      }

      // Blockquotes or formulas
      if (trimmed.startsWith(">")) {
        return (
          <blockquote key={idx} className="border-l-4 border-indigo-400 bg-indigo-50/40 px-3 py-2 my-2 rounded-r-lg text-sm italic text-gray-650">
            {parseInlineStyles(trimmed.replace(/^>\s*/, ""))}
          </blockquote>
        );
      }

      // Normal paragraph
      if (trimmed.length === 0) {
        return <div key={idx} className="h-2" />;
      }

      return (
        <p key={idx} className="text-sm md:text-base leading-relaxed text-gray-700 text-justify mb-2">
          {parseInlineStyles(line)}
        </p>
      );
    });
  };

  // Formatter for inline **bold** and `code` spans
  const parseInlineStyles = (s: string) => {
    const parts = [];
    let currentIdx = 0;

    // Regex matching either **bold** or `code`
    const regex = /(\*\*|`)(.*?)\1/g;
    let match;

    while ((match = regex.exec(s)) !== null) {
      const matchStart = match.index;
      const type = match[1];
      const content = match[2];

      // Push proceeding plain text
      if (matchStart > currentIdx) {
        parts.push(s.substring(currentIdx, matchStart));
      }

      // Push styled element
      if (type === "**") {
        parts.push(<strong key={matchStart} className="font-semibold text-gray-900">{content}</strong>);
      } else {
        parts.push(<code key={matchStart} className="px-1.5 py-0.5 bg-gray-150 font-mono text-xs text-rose-600 rounded bg-gray-100">{content}</code>);
      }

      currentIdx = regex.lastIndex;
    }

    if (currentIdx < s.length) {
      parts.push(s.substring(currentIdx));
    }

    return parts.length > 0 ? parts : s;
  };

  return (
    <div id="chat-assistant-root" className="flex flex-col h-full bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm">


      {/* Message Area Stream */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-gray-50/50">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col justify-start items-center text-center p-4 space-y-4 overflow-y-auto">
            <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center border border-blue-100 shrink-0 mt-4">
              <HelpCircle className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-gray-800">Need help understanding the paper?</h3>
              <p className="text-xs text-gray-500 max-w-xs mt-1 leading-relaxed">
                Ask any questions about hypotheses, mathematical equations, experimental results, or practical implications. The AI assistant will provide structured, factual answers based on the paper.
              </p>
            </div>

            {/* Quick buttons suggestions helper */}
            <div className="w-full max-w-sm pt-2 space-y-2">
              <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest text-left pl-1">Suggested Questions:</div>
              <div className="flex flex-col gap-2">
                {presetPrompts.map((prompt, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => onSendMessage(prompt)}
                    className="w-full px-3 py-2.5 bg-white border border-gray-200 hover:bg-blue-50 hover:border-blue-200 rounded-xl text-xs text-left text-gray-700 font-medium transition-all shadow-2xs flex items-start space-x-2"
                  >
                    <CornerDownRight className="w-3.5 h-3.5 text-blue-500 flex-shrink-0 mt-0.5" />
                    <span className="line-clamp-2 leading-relaxed">{prompt}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Recent Questions section (only shown if questions were actually asked for this paper) */}
            {historyItems.length > 0 && (
              <div className="w-full max-w-sm pt-4 space-y-2 border-t border-gray-150">
                <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest text-left pl-1">Recent Questions:</div>
                <div className="flex flex-col gap-1.5">
                  {historyItems.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => onSelectHistoryItem ? onSelectHistoryItem(item) : onSendMessage(item.question)}
                      className="w-full px-3 py-2 bg-slate-50 border border-slate-200 hover:bg-slate-100 hover:border-slate-300 rounded-xl text-xs text-left text-gray-600 font-medium transition-colors shadow-2xs flex items-center space-x-2"
                    >
                      <Clock className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
                      <span className="truncate flex-1" title={item.question}>{item.question}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((msg) => {
              const isAi = msg.sender === "ai";
              return (
                <div key={msg.id} className={`flex flex-col ${isAi ? "items-start" : "items-end"}`}>
                  {/* Msg Context block details if passage linked */}
                  {msg.passageReference && (
                    <div className="max-w-[85%] mb-1 px-3 py-1.5 bg-gray-200/60 rounded-xl border border-gray-300 text-[11px] text-gray-500 italic font-mono flex items-center space-x-1 shadow-2xs">
                      <span>Referenced text: &quot;{msg.passageReference.text.substring(0, 45)}...&quot;</span>
                    </div>
                  )}

                  <div className={`max-w-[90%] md:max-w-[85%] rounded-2xl px-4 py-3.5 shadow-2xs ${isAi
                    ? "bg-white text-gray-800 border border-gray-200"
                    : "bg-blue-500 text-white rounded-tr-none"
                    }`}>
                    <div className="space-y-1 font-sans">
                      {isAi ? renderFormattedMessage(msg.text) : (
                        <p className="text-sm md:text-base leading-relaxed text-justify break-words">{msg.text}</p>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}

            {/* AI Typing Loader indicator */}
            {isLoading && (
              <div className="flex items-start">
                <div className="max-w-[85%] bg-white border border-gray-200 rounded-2xl px-4 py-3.5 shadow-2xs">
                  <div className="flex items-center space-x-1.5 mb-1">
                    <div className="p-0.5 bg-blue-50 text-blue-600 rounded-md">
                      <Cpu className="w-3.5 h-3.5" />
                    </div>
                    <span className="text-[10px] font-bold text-blue-600 tracking-wider">ResearchOS</span>
                  </div>
                  <div className="flex items-center space-x-1.5 py-2">
                    <RefreshCw className="w-3.5 h-3.5 text-blue-500 animate-spin" />
                    <span className="text-xs text-gray-500 italic">AI is analyzing & retrieving evidence from the paper...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={chatBottomRef} />
          </div>
        )}
      </div>

      {/* Floating Passage selection linker widget */}
      {passageReference && (
        <div className="px-4 py-2.5 bg-amber-50 border-t border-amber-100 flex items-center justify-between text-xs text-amber-900 shadow-sm animate-fade-in font-sans">
          <div className="flex items-center space-x-1.5 truncate">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
            </span>
            <span className="font-semibold text-[11px] bg-amber-100 px-1.5 py-0.5 rounded text-amber-800 flex-shrink-0 font-sans">Selection</span>
            <span className="truncate italic text-gray-700">“{passageReference.text}”</span>
          </div>
          <button
            onClick={onClearPassage}
            className="text-amber-600 hover:text-amber-800 p-1 rounded hover:bg-amber-100/50 bg-transparent animate-pulse"
            title="Unlink passage"
          >
            <XCircle className="w-4 h-4 cursor-pointer" />
          </button>
        </div>
      )}

      {/* Input area */}
      <div className="p-4 border-t border-gray-200 bg-white">
        <form onSubmit={onSubmit} className="flex space-x-2">
          <input
            id="chat-input-text"
            type="text"
            placeholder={
              passageReference
                ? "Ask about the linked passage..."
                : "Ask about theory, equations, or scientific findings..."
            }
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isLoading}
            className="flex-1 bg-gray-55/70 text-gray-800 border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400"
          />
          <button
            id="btn-send-message"
            type="submit"
            disabled={!inputText.trim() || isLoading}
            className="p-3 bg-blue-500 hover:bg-blue-600 text-white rounded-xl transition-all disabled:opacity-50 disabled:hover:bg-blue-500 border-none flex items-center justify-center shadow-md cursor-pointer"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>

        {/* Helper footer */}
        <div className="flex items-center justify-between mt-2 px-1">
          <p className="text-[10px] text-gray-400 text-center flex-1">
            Grounded directly on the published text of the selected research paper.
          </p>
          {messages.length > 0 && (
            <button
              onClick={onClearHistory}
              className="text-gray-400 hover:text-rose-650 transition-colors border-none bg-transparent flex items-center justify-center p-1 rounded cursor-pointer"
              title="Clear conversation history"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
