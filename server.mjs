import "dotenv/config";
import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import OpenAI from "openai";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = Number(process.env.PORT || 3000);
const MODEL = process.env.MODEL || process.env.OPENAI_MODEL || "gpt-4o-mini";
const VECTOR_STORE_NAME = process.env.VECTOR_STORE_NAME || "ApoMind Knowledge Base";
const KNOWLEDGE_DOC_PATH = path.resolve(__dirname, process.env.KNOWLEDGE_FILE || "./knowledge.docx");
const SYSTEM_PROMPT =
  "Είσαι ο ApoMind Ψηφιακός Βοηθός. Απαντάς αποκλειστικά με βάση τα αποσπάσματα που θα ανακτηθούν από το έγγραφο γνώσης μέσω file search. " +
  "Αν το έγγραφο δεν περιέχει σαφή απάντηση, απαντάς ακριβώς: 'Δεν βρήκα σχετική πληροφορία στο έγγραφο γνώσης.' " +
  "Μην χρησιμοποιείς γενικές γνώσεις, μην κάνεις υποθέσεις, μην συμπληρώνεις κενά. " +
  "Να απαντάς στα ελληνικά, σύντομα, καθαρά και επαγγελματικά.";
app.get("/", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});
if (!process.env.OPENAI_API_KEY) {
  console.warn("[WARN] Missing OPENAI_API_KEY in environment.");
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
let inMemoryVectorStoreId = process.env.OPENAI_VECTOR_STORE_ID?.trim() || "";
let bootstrapPromise = null;

app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

function getFallbackReply() {
  return "Δεν βρήκα σχετική πληροφορία στο έγγραφο γνώσης.";
}

function extractTextOutput(response) {
  const text = response?.output_text?.trim();
  return text || getFallbackReply();
}

async function ensureKnowledgeFileExists() {
  if (!fs.existsSync(KNOWLEDGE_DOC_PATH)) {
    throw new Error(`Δεν βρέθηκε το knowledge αρχείο στο path: ${KNOWLEDGE_DOC_PATH}`);
  }
}

async function createFreshVectorStore() {
  await ensureKnowledgeFileExists();

  const vectorStore = await client.vectorStores.create({
    name: VECTOR_STORE_NAME,
  });

  const uploadedFile = await client.files.create({
    file: fs.createReadStream(KNOWLEDGE_DOC_PATH),
    purpose: "assistants",
  });

  await client.vectorStores.files.createAndPoll(vectorStore.id, {
    file_id: uploadedFile.id,
  });

  inMemoryVectorStoreId = vectorStore.id;
  return vectorStore.id;
}

async function getVectorStoreId({ allowBootstrap = false } = {}) {
  const envStoreId = process.env.OPENAI_VECTOR_STORE_ID?.trim();
  if (envStoreId) {
    inMemoryVectorStoreId = envStoreId;
    return envStoreId;
  }

  if (inMemoryVectorStoreId) {
    return inMemoryVectorStoreId;
  }

  if (!allowBootstrap) {
    return "";
  }

  if (!bootstrapPromise) {
    bootstrapPromise = createFreshVectorStore().finally(() => {
      bootstrapPromise = null;
    });
  }

  return bootstrapPromise;
}

app.get("/api/health", async (_req, res) => {
  const vectorStoreId = await getVectorStoreId();

  res.json({
    ok: true,
    model: MODEL,
    has_api_key: Boolean(process.env.OPENAI_API_KEY),
    knowledge_file_exists: fs.existsSync(KNOWLEDGE_DOC_PATH),
    vector_store_id: vectorStoreId || null,
    using_env_vector_store: Boolean(process.env.OPENAI_VECTOR_STORE_ID),
  });
});

app.get("/api/knowledge/status", async (_req, res) => {
  const vectorStoreId = await getVectorStoreId();

  res.json({
    ok: true,
    knowledge_file_exists: fs.existsSync(KNOWLEDGE_DOC_PATH),
    knowledge_file_path: path.basename(KNOWLEDGE_DOC_PATH),
    vector_store_id: vectorStoreId || null,
    using_env_vector_store: Boolean(process.env.OPENAI_VECTOR_STORE_ID),
  });
});

app.post("/api/bootstrap-knowledge", async (_req, res) => {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "Λείπει το OPENAI_API_KEY." });
    }

    const vectorStoreId = await getVectorStoreId({ allowBootstrap: true });

    return res.json({
      ok: true,
      message:
        process.env.OPENAI_VECTOR_STORE_ID
          ? "Χρησιμοποιείται το υπάρχον OPENAI_VECTOR_STORE_ID από τα environment variables."
          : "Το knowledge.docx ανέβηκε και έγινε index επιτυχώς. Αποθήκευσε το vector_store_id στο OPENAI_VECTOR_STORE_ID στο Vercel.",
      vector_store_id: vectorStoreId,
      using_env_vector_store: Boolean(process.env.OPENAI_VECTOR_STORE_ID),
    });
  } catch (error) {
    console.error("Bootstrap knowledge error:", error);
    return res.status(500).json({
      error: "Αποτυχία bootstrap knowledge file.",
      details: error?.message || "Unknown error",
    });
  }
});

app.post("/api/chat", async (req, res) => {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "Λείπει το OPENAI_API_KEY." });
    }

    const message = String(req.body?.message || "").trim();
    if (!message) {
      return res.status(400).json({ error: "Το message είναι υποχρεωτικό." });
    }

    const vectorStoreId = await getVectorStoreId({ allowBootstrap: true });
    if (!vectorStoreId) {
      return res.status(500).json({
        error: "Δεν μπόρεσα να αρχικοποιήσω τη knowledge βάση.",
      });
    }

    const response = await client.responses.create({
      model: MODEL,
      input: [
        {
          role: "system",
          content: [{ type: "input_text", text: SYSTEM_PROMPT }],
        },
        {
          role: "user",
          content: [{ type: "input_text", text: message }],
        },
      ],
      tools: [
        {
          type: "file_search",
          vector_store_ids: [vectorStoreId],
          max_num_results: 5,
        },
      ],
      include: ["file_search_call.results"],
    });

    return res.json({
      reply: extractTextOutput(response),
      vector_store_id: vectorStoreId,
    });
  } catch (error) {
    console.error("Chat error:", error);
    return res.status(500).json({
      error: "Σφάλμα στον server.",
      details: error?.message || "Unknown error",
    });
  }
});

// ─── QUIZ ENDPOINTS ───────────────────────────────────────────────────────────

const QUIZ_GENERATE_PROMPT = `
Είσαι ο ApoMind Ψηφιακός Βοηθός. Βασισμένος ΑΠΟΚΛΕΙΣΤΙΚΑ στα αποσπάσματα από το έγγραφο γνώσης, 
δημιούργησε ένα quiz με ακριβώς {COUNT} ερωτήσεις πολλαπλής επιλογής.

ΚΑΝΟΝΕΣ:
- Κάθε ερώτηση πρέπει να έχει ακριβώς 4 επιλογές (A, B, C, D)
- Μόνο μία επιλογή είναι σωστή
- Οι ερωτήσεις πρέπει να βασίζονται ΜΟΝΟ στο περιεχόμενο του εγγράφου
- Αν το έγγραφο δεν έχει αρκετό υλικό, δημιούργησε όσες μπορείς

Απάντησε ΜΟΝΟ με valid JSON, χωρίς markdown, χωρίς backticks, ακριβώς σε αυτή τη μορφή:
{
  "questions": [
    {
      "id": 1,
      "question": "Κείμενο ερώτησης;",
      "options": {
        "A": "Επιλογή Α",
        "B": "Επιλογή Β",
        "C": "Επιλογή Γ",
        "D": "Επιλογή Δ"
      },
      "correct": "A",
      "explanation": "Σύντομη εξήγηση γιατί είναι σωστή η απάντηση."
    }
  ]
}
`;

const QUIZ_EVALUATE_PROMPT = `
Είσαι ο ApoMind Ψηφιακός Βοηθός. Ο χρήστης μόλις ολοκλήρωσε ένα quiz.
Δες τα αποτελέσματα και δώσε ενθαρρυντικό, εποικοδομητικό σχόλιο στα ελληνικά.
Αναφέρσου στον αριθμό σωστών/λανθασμένων και δώσε μια σύντομη ανάλυση (3-4 προτάσεις).
`;

app.post("/api/quiz/generate", async (req, res) => {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "Λείπει το OPENAI_API_KEY." });
    }

    const count = Math.min(Math.max(parseInt(req.body?.count || 5), 3), 10);

    const vectorStoreId = await getVectorStoreId({ allowBootstrap: true });
    if (!vectorStoreId) {
      return res.status(500).json({ error: "Δεν μπόρεσα να αρχικοποιήσω τη knowledge βάση." });
    }

    const prompt = QUIZ_GENERATE_PROMPT.replace("{COUNT}", count);

    const response = await client.responses.create({
      model: MODEL,
      input: [
        {
          role: "system",
          content: [{ type: "input_text", text: prompt }],
        },
        {
          role: "user",
          content: [{ type: "input_text", text: `Δημιούργησε ${count} ερωτήσεις quiz από το έγγραφο γνώσης.` }],
        },
      ],
      tools: [
        {
          type: "file_search",
          vector_store_ids: [vectorStoreId],
          max_num_results: 10,
        },
      ],
      include: ["file_search_call.results"],
    });

    const rawText = response?.output_text?.trim() || "";

    // Strip markdown code fences if present
    const cleaned = rawText.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "").trim();

    let quizData;
    try {
      quizData = JSON.parse(cleaned);
    } catch (e) {
      console.error("Quiz JSON parse error:", e, "Raw:", rawText);
      return res.status(500).json({ error: "Αποτυχία δημιουργίας quiz. Δοκίμασε ξανά." });
    }

    return res.json({ ok: true, quiz: quizData });
  } catch (error) {
    console.error("Quiz generate error:", error);
    return res.status(500).json({
      error: "Σφάλμα κατά τη δημιουργία quiz.",
      details: error?.message || "Unknown error",
    });
  }
});

app.post("/api/quiz/evaluate", async (req, res) => {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "Λείπει το OPENAI_API_KEY." });
    }

    const { score, total, wrong_topics } = req.body || {};

    const userSummary = `Ο χρήστης απάντησε σωστά σε ${score} από τις ${total} ερωτήσεις.${
      wrong_topics?.length ? ` Δυσκολεύτηκε στα εξής θέματα: ${wrong_topics.join(", ")}.` : ""
    }`;

    const response = await client.responses.create({
      model: MODEL,
      input: [
        {
          role: "system",
          content: [{ type: "input_text", text: QUIZ_EVALUATE_PROMPT }],
        },
        {
          role: "user",
          content: [{ type: "input_text", text: userSummary }],
        },
      ],
    });

    return res.json({ ok: true, feedback: response?.output_text?.trim() || "Μπράβο για τη συμμετοχή!" });
  } catch (error) {
    console.error("Quiz evaluate error:", error);
    return res.status(500).json({ error: "Σφάλμα αξιολόγησης.", details: error?.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────

if (process.env.VERCEL !== "1") {
  app.listen(port, () => {
    console.log(`ApoMind backend running on http://localhost:${port}`);
  });
}

export default app;
