/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "15mb" }));

// Lazy initializer for Gemini client to prevent crash on empty env key
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const key = process.env.GEMINI_API_KEY;
    if (!key) {
      throw new Error("Không tìm thấy mã đăng ký GEMINI_API_KEY. Vui lòng thiết lập khóa trong bảng Secrets.");
    }
    aiClient = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// ----------------------------------------------------
// API ROUTES
// ----------------------------------------------------

/**
 * Endpoint for Chatbot to answer question based on selected paper context
 */
app.post("/api/chat", async (req, res) => {
  try {
    const { message, history, paper, passageReference } = req.body;

    if (!message || !paper) {
      res.status(400).json({ error: "Thiếu dữ liệu câu hỏi (message) hoặc tài liệu (paper)." });
      return;
    }

    const ai = getGeminiClient();

    // Prepare system instruction providing full paper context
    const sectionsText = paper.sections
      ? paper.sections.map((s: any) => `### Khối: ${s.title}\nNội dung: ${s.content}`).join("\n\n")
      : "";

    const keyFindingsText = paper.keyFindings
      ? paper.keyFindings.map((f: string) => `- ${f}`).join("\n")
      : "";

    let systemInstruction = `Bạn là "SciMind AI" - trợ lý nghiên cứu khoa học chuyên sâu, có chuyên môn cao trong việc phân tích, tóm tắt và giải nghĩa các bài báo khoa học phức tạp.
Người dùng hiện đang đọc bài báo khoa học sau đây trong giao diện Split-Screen:

--- BÀI BÁO ---
Tiêu đề: ${paper.title}
Tác giả: ${paper.authors} (${paper.year || "Không rõ năm"})
DOI / Tạp chí: ${paper.doi || "N/A"} / ${paper.journal || "N/A"}

Tóm tắt (Abstract):
${paper.abstract}

Nội dung chi tiết các phần:
${sectionsText}

Kết luận & Phát hiện cốt lõi (Key Findings):
${keyFindingsText}
--------------`;

    if (passageReference) {
      systemInstruction += `\n\nLƯU Ý ĐẶC BIỆT: Người dùng vừa bôi đen / chọn một đoạn văn cụ thể trong bài báo và gửi yêu cầu giải thích về nó:
Đoạn văn được tham chiếu: "${passageReference.text}" trong phần "${passageReference.sectionTitle}".
Hãy giải thích đoạn văn bôi đen này thật dễ hiểu, cắt nghĩa từng ý và liên hệ trực tiếp tới bối cảnh tổng thể của bài báo nghiên cứu này.`;
    }

    systemInstruction += `\n\nHướng dẫn ứng xử & Định dạng câu trả lời:
1. Hãy trả lời câu hỏi trực tiếp bằng lối diễn đạt khoa học chuyên sâu, rõ ràng, giàu tính học thuật nhưng dễ tiếp cận đối với sinh viên hay nghiên cứu sinh.
2. Nếu phân tích công thức toán học, hãy lý giải từng biến số một cách tỉ mỉ. Nếu đưa ra so sánh hoặc ví dụ đời thực để minh họa thuật toán, điều đó rất được khuyến khích.
3. Luôn sử dụng Markdown (như lập danh sách, in đậm chữ quan trọng, bảng so sánh hoặc khối code chứa công thức dạng LaTeX) để giữ định dạng văn bản sang trọng, rõ ràng.
4. Ưu tiên câu trả lời bằng tiếng Việt trừ phi người dùng đặt câu hỏi bằng tiếng Anh hay ngôn ngữ khác. Hãy lịch lãm, khách quan, trung thực (nếu thông tin không nằm trong bài báo và bạn suy diễn từ kiến thức chung, hãy nói rõ là "theo kiến thức nền của tôi").`;

    // Map conversation history into candidates structure
    // history: Array of { role: 'user' | 'model', text: string }
    const contents = [];
    if (history && history.length > 0) {
      for (const h of history) {
        contents.push({
          role: h.sender === "user" ? "user" : "model",
          parts: [{ text: h.text }],
        });
      }
    }

    // Append current user prompt
    contents.push({
      role: "user",
      parts: [
        {
          text: passageReference
            ? `Hãy giải thích kỹ đoạn văn tôi chọn này: "${passageReference.text}"\n\nCâu hỏi kèm theo: ${message}`
            : message,
        },
      ],
    });

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: contents,
      config: {
        systemInstruction,
        temperature: 0.5,
      },
    });

    res.json({ text: response.text });
  } catch (error: any) {
    console.error("Lỗi trong quá trình trò chuyện với Gemini:", error);
    res.status(500).json({ error: error.message || "Lỗi máy chủ nội bộ trong quá trình trao đổi với AI." });
  }
});

/**
 * Advanced endpoint that parses raw scientific paper text into organized sections using Gemini
 */
app.post("/api/analyze-paper", async (req, res) => {
  try {
    const { rawText, title } = req.body;

    if (!rawText || rawText.trim().length < 50) {
      res.status(400).json({ error: "Vui lòng nhập văn bản bài báo tối thiểu 50 ký tự." });
      return;
    }

    const ai = getGeminiClient();

    const analysisPrompt = `Bạn là một mô hình AI phân tích cấu trúc bài báo khoa học xuất sắc.
Hãy phân tích tài liệu văn bản thô của một bài báo nghiên cứu khoa học dưới đây, sau đó bóc tách cấu trúc và trả về một định dạng JSON hoàn hảo mô tả bài báo đó.

Văn bản thô của bài báo:
"""
${rawText}
"""

Hãy tự suy luận hoặc trích xuất:
1. Title (Tiêu đề bài báo - nếu không tìm thấy hãy đặt một tiêu đề mô tả chính xác từ nội dung, hoặc dùng tiêu đề tạm thô sau nếu thấy hợp lý: "${title || ""}")
2. Authors (Danh sách tác giả - nếu không có hãy ghi là "Đang cập nhật")
3. Year (Năm xuất bản - dạng chuỗi, ví dụ "2024", nếu không rõ ghi "2024")
4. Abstract (Tóm tắt ngắn gọn khoảng 150-200 từ bằng Tiếng Việt súc tích, chuyên sâu, nói rõ mục tiêu, phương pháp và kết quả)
5. Sections (Danh sách các phân mục kết cấu của bài báo nghiên cứu, tối thiểu gồm các khối chính như Introduction, Methodology/Proposed Method, Experiments/Results, Discussion/Conclusion. Viết nội dung bằng Tiếng Việt học thuật súc tích, dịch nghĩa các điểm mấu chốt để người đọc tiện nghiên cứu)
6. Metrics (Nhận định độ đột phá sáng tạo từ 0-100, mức độ phức tạp ["Cơ bản", "Trung bình", "Chuyên sâu"], thời gian đọc dự kiến theo phút, và số lượng trích dẫn giả định hoặc ước tính dựa trên tầm quan trọng)
7. Key findings (3-5 phát hiện cốt lõi hoặc ý tưởng đóng góp chính)
8. Glossary (Bảng chú giải thuật ngữ quan trọng nhất gồm 3-5 từ khóa học thuật được sử dụng rộng rãi kèm theo định nghĩa chi tiết bằng tiếng Việt của chúng)`;

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: analysisPrompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: {
              type: Type.STRING,
              description: "Tiêu đề bài báo khoa học dịch nghĩa tiếng Việt sắc sảo hoặc tiêu đề gốc tiếng Anh nếu thông dụng.",
            },
            authors: {
              type: Type.STRING,
              description: "Tên các tác giả bài báo.",
            },
            year: {
              type: Type.STRING,
              description: "Năm công bố bài báo.",
            },
            abstract: {
              type: Type.STRING,
              description: "Bản tóm tắt bằng Tiếng Việt học thuật, súc tích diễn tả giá trị bài báo.",
            },
            sections: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING, description: "Tiêu đề phần, ví dụ '1. Introduction (Giới thiệu)'" },
                  content: { type: Type.STRING, description: "Nội dung phân tích/trình bày tóm tắt tiếng Việt của chương này, giữ nguyên các công thức quan trọng." },
                },
                required: ["title", "content"],
              },
            },
            metrics: {
              type: Type.OBJECT,
              properties: {
                novelty: { type: Type.INTEGER, description: "Điểm đột phá từ 0 đến 100" },
                complexity: { type: Type.STRING, description: "Mức độ phức tạp: 'Cơ bản', 'Trung bình' hoặc 'Chuyên sâu'" },
                readingTime: { type: Type.INTEGER, description: "Thời gian đọc dự kiến (phút)" },
                citations: { type: Type.INTEGER, description: "Lượng trích dẫn ước lượng" },
              },
              required: ["novelty", "complexity", "readingTime", "citations"],
            },
            keyFindings: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Đóng góp quan trọng hoặc kết quả chính.",
            },
            glossary: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  term: { type: Type.STRING, description: "Thuật ngữ học thuật chuyên sâu bằng tiếng Anh hoặc tiếng Việt" },
                  definition: { type: Type.STRING, description: "Giải nghĩa từ vựng này trong bối cảnh bài báo bằng tiếng Việt" },
                },
                required: ["term", "definition"],
              },
            },
          },
          required: ["title", "authors", "year", "abstract", "sections", "metrics", "keyFindings", "glossary"],
        },
      },
    });

    const parsedJson = JSON.parse(response.text.trim());
    res.json(parsedJson);
  } catch (error: any) {
    console.error("Lỗi phân tích tài liệu bài báo:", error);
    res.status(500).json({ error: error.message || "Lỗi máy chủ khi parse tài liệu." });
  }
});

// ----------------------------------------------------
// VITE OR STATIC SERVING MIDDLEWARE
// ----------------------------------------------------
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    console.log("Configuring Vite Middleware in the Development Environment...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("Serving static content in a Production environment...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running at http://0.0.0.0:${PORT}`);
  });
}

startServer();
