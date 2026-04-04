import os
import re
import threading
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs('uploads', exist_ok=True)


_embeddings_cache = None
_embeddings_lock = threading.Lock()

def get_embeddings():
    """Return a cached OllamaEmbeddings instance (created once, reused always)."""
    global _embeddings_cache
    with _embeddings_lock:
        if _embeddings_cache is None:
            from langchain_community.embeddings import OllamaEmbeddings
            
            _embeddings_cache = OllamaEmbeddings(model="nomic-embed-text")
        return _embeddings_cache

rag_state = {
    "db": None,
    "rag_chain": None,
    "loaded_file": None,
    "num_chunks": 0,
    "status": "idle",     
    "progress_msg": "",
    "error": ""
}


def init_rag_background(pdf_path, filename):
    """Run the full RAG pipeline in a background thread."""
    try:
        rag_state["status"] = "processing"
        rag_state["error"] = ""

        
        rag_state["progress_msg"] = "Loading PDF pages..."
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        
        rag_state["progress_msg"] = "Cleaning & splitting text..."
        def clean_text(text):
            text = text.replace("\n", " ")
            text = re.sub(r'\s+', ' ', text)
            return text

        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        
        rag_state["progress_msg"] = f"Embedding {len(chunks)} chunks via Ollama..."
        embeddings = get_embeddings()

        
        rag_state["progress_msg"] = "Building FAISS vector index..."
        from langchain_community.vectorstores import FAISS
        db = FAISS.from_documents(chunks, embeddings, normalize_L2=True)

        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": 3}
        )

       
        rag_state["progress_msg"] = "Initialising RAG chain..."
        from langchain_community.llms import Ollama
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_template("""You are an AI assistant specialised in answering questions about documents.
Use ONLY the provided context to answer.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:""")

        llm = Ollama(model="gemma:2b", temperature=0.2, num_predict=500)
        parser = StrOutputParser()

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | parser
        )

        
        rag_state["db"] = db
        rag_state["rag_chain"] = rag_chain
        rag_state["loaded_file"] = filename
        rag_state["num_chunks"] = len(chunks)
        rag_state["progress_msg"] = "Ready!"
        rag_state["status"] = "ready"

    except Exception as e:
        rag_state["status"] = "error"
        rag_state["error"] = str(e)
        rag_state["progress_msg"] = ""


HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>DocMind — PDF Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap" rel="stylesheet"/>
<style>
  :root {
    --ink: #0d0d0d; --paper: #f5f0e8; --accent: #c8401a;
    --accent2: #1a5fc8; --muted: #6b6560; --border: #d4cec4;
    --success: #1a7c3e;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "DM Mono", monospace; background: var(--paper); color: var(--ink); min-height: 100vh; overflow-x: hidden; }

  header { border-bottom: 2px solid var(--ink); padding: 18px 48px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; background: var(--paper); z-index: 100; }
  .logo { font-family: "Syne", sans-serif; font-weight: 800; font-size: 1.4rem; letter-spacing: -0.03em; }
  .logo span { color: var(--accent); }
  .status-pill { font-size: 0.72rem; padding: 4px 12px; border: 1.5px solid var(--border); border-radius: 99px; color: var(--muted); transition: all 0.3s; }
  .status-pill.ready { border-color: var(--success); color: var(--success); }
  .status-pill.processing { border-color: var(--accent); color: var(--accent); }

  .main { display: grid; grid-template-columns: 340px 1fr; min-height: calc(100vh - 61px); }
  .sidebar { border-right: 2px solid var(--ink); padding: 36px 28px; display: flex; flex-direction: column; gap: 28px; }
  .section-label { font-family: "Syne", sans-serif; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }

  .upload-zone { border: 2px dashed var(--border); border-radius: 6px; padding: 28px 20px; text-align: center; cursor: pointer; transition: all 0.25s; position: relative; background: white; }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--accent); background: #fdf5f0; }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  .upload-icon { font-size: 2rem; margin-bottom: 10px; }
  .upload-text { font-size: 0.78rem; color: var(--muted); line-height: 1.6; }
  .upload-text strong { color: var(--ink); font-weight: 400; }

  .btn { width: 100%; padding: 12px; font-family: "Syne", sans-serif; font-weight: 700; font-size: 0.82rem; letter-spacing: 0.06em; text-transform: uppercase; border: 2px solid var(--ink); background: var(--ink); color: var(--paper); border-radius: 4px; cursor: pointer; transition: all 0.2s; }
  .btn:hover:not(:disabled) { background: var(--accent); border-color: var(--accent); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .file-info { background: white; border: 1.5px solid var(--border); border-radius: 6px; padding: 14px 16px; font-size: 0.75rem; display: none; }
  .file-info.show { display: block; }
  .file-info .fname { font-weight: 600; word-break: break-all; margin-bottom: 6px; }
  .file-info .fmeta { color: var(--muted); }
  .file-info .fmeta .chunks { color: var(--success); font-weight: 600; }

  .progress-wrap { display: none; background: white; border: 1.5px solid var(--border); border-radius: 6px; padding: 14px 16px; }
  .progress-wrap.show { display: block; }
  .progress-bar-outer { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 10px; }
  .progress-bar-inner { height: 100%; background: var(--accent); width: 40%; }
  .progress-bar-inner.running { animation: slide 1.6s ease-in-out infinite; }
  @keyframes slide { 0% { margin-left: -40%; } 100% { margin-left: 100%; } }
  .progress-bar-inner.done { width: 100% !important; animation: none; background: var(--success); transition: width 0.4s; }
  .progress-msg { font-size: 0.72rem; color: var(--muted); }

  .suggestions { display: flex; flex-direction: column; gap: 8px; }
  .suggestion-btn { text-align: left; padding: 10px 14px; font-family: "DM Mono", monospace; font-size: 0.73rem; border: 1.5px solid var(--border); border-radius: 4px; background: white; cursor: pointer; color: var(--ink); transition: all 0.2s; line-height: 1.5; }
  .suggestion-btn:hover { border-color: var(--accent2); color: var(--accent2); background: #f0f5ff; }

  .chat-panel { display: flex; flex-direction: column; height: calc(100vh - 61px); }
  .chat-header { padding: 20px 36px; border-bottom: 1.5px solid var(--border); font-family: "Syne", sans-serif; font-size: 0.85rem; font-weight: 600; color: var(--muted); }
  .messages { flex: 1; overflow-y: auto; padding: 32px 36px; display: flex; flex-direction: column; gap: 24px; scroll-behavior: smooth; }
  .empty-state { margin: auto; text-align: center; max-width: 360px; color: var(--muted); }
  .empty-state .big-icon { font-size: 3.5rem; margin-bottom: 16px; opacity: 0.4; }
  .empty-state p { font-size: 0.82rem; line-height: 1.8; }

  .msg { display: flex; gap: 14px; animation: slide-in 0.3s ease; max-width: 820px; }
  @keyframes slide-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg-avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; flex-shrink: 0; border: 1.5px solid var(--border); }
  .msg.user .msg-avatar { background: var(--ink); color: var(--paper); border-color: var(--ink); }
  .msg.bot .msg-avatar { background: white; }
  .msg-body { background: white; border: 1.5px solid var(--border); border-radius: 6px; padding: 14px 18px; font-size: 0.82rem; line-height: 1.9; max-width: 680px; white-space: pre-wrap; word-break: break-word; }
  .msg.user .msg-body { background: var(--ink); color: var(--paper); border-color: var(--ink); }
  .msg-body.thinking { color: var(--muted); font-style: italic; display: flex; align-items: center; gap: 8px; }
  .dots { display: inline-flex; gap: 4px; }
  .dots span { width: 5px; height: 5px; border-radius: 50%; background: var(--muted); animation: dot-bounce 1.2s ease-in-out infinite; }
  .dots span:nth-child(2) { animation-delay: 0.2s; }
  .dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes dot-bounce { 0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }

  .input-area { border-top: 2px solid var(--ink); padding: 20px 36px; display: flex; gap: 12px; align-items: flex-end; }
  .input-wrap { flex: 1; }
  textarea { width: 100%; padding: 13px 16px; font-family: "DM Mono", monospace; font-size: 0.82rem; border: 1.5px solid var(--border); border-radius: 6px; background: white; color: var(--ink); resize: none; outline: none; transition: border-color 0.2s; min-height: 50px; max-height: 140px; overflow-y: auto; line-height: 1.6; }
  textarea:focus { border-color: var(--ink); }
  textarea:disabled { opacity: 0.5; cursor: not-allowed; }
  .send-btn { width: 50px; height: 50px; border-radius: 6px; border: 2px solid var(--ink); background: var(--ink); color: var(--paper); cursor: pointer; font-size: 1.1rem; transition: all 0.2s; flex-shrink: 0; }
  .send-btn:hover:not(:disabled) { background: var(--accent); border-color: var(--accent); }
  .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .toast { position: fixed; bottom: 24px; right: 24px; padding: 12px 20px; border-radius: 6px; font-size: 0.78rem; transform: translateY(80px); opacity: 0; transition: all 0.3s; z-index: 999; max-width: 340px; line-height: 1.5; }
  .toast.show { transform: translateY(0); opacity: 1; }
  .toast.error { background: #3d0d05; color: #ffb3a0; border: 1px solid #c8401a; }
  .toast.success { background: #0d2e1a; color: #90e6b4; border: 1px solid #1a7c3e; }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <div class="logo">Doc<span>Mind</span></div>
  <div class="status-pill" id="statusPill">No document loaded</div>
</header>

<div class="main">
  <aside class="sidebar">
    <div>
      <div class="section-label">Upload Document</div>
      <div class="upload-zone" id="dropZone">
        <input type="file" id="fileInput" accept=".pdf"/>
        <div class="upload-icon">📄</div>
        <div class="upload-text"><strong>Drop a PDF here</strong><br/>or click to browse</div>
      </div>
    </div>

    <div class="file-info" id="fileInfo">
      <div class="fname" id="fileName">—</div>
      <div class="fmeta">Chunks: <span class="chunks" id="chunkCount">—</span> &nbsp;|&nbsp; gemma:2b</div>
    </div>

    <div class="progress-wrap" id="progressWrap">
      <div class="progress-bar-outer">
        <div class="progress-bar-inner running" id="progressBar"></div>
      </div>
      <div class="progress-msg" id="progressMsg">Starting...</div>
    </div>

    <button class="btn" id="uploadBtn" onclick="uploadPDF()" disabled>Process Document</button>

    <div>
      <div class="section-label">Suggested Questions</div>
      <div class="suggestions">
        <button class="suggestion-btn" onclick="useSuggestion(this)">What is the main topic of this document?</button>
        <button class="suggestion-btn" onclick="useSuggestion(this)">Summarise the key findings.</button>
        <button class="suggestion-btn" onclick="useSuggestion(this)">Explain the methodology used.</button>
        <button class="suggestion-btn" onclick="useSuggestion(this)">What conclusions are drawn?</button>
      </div>
    </div>
  </aside>

  <div class="chat-panel">
    <div class="chat-header" id="chatHeader">Upload a PDF to begin querying →</div>
    <div class="messages" id="messages">
      <div class="empty-state" id="emptyState">
        <div class="big-icon">🔍</div>
        <p>Load a PDF from the sidebar.<br/>DocMind indexes it locally using FAISS + Gemma:2b via Ollama.</p>
      </div>
    </div>
    <div class="input-area">
      <div class="input-wrap">
        <textarea id="queryInput" placeholder="Ask a question about your document..." rows="1"
          disabled onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      </div>
      <button class="send-btn" id="sendBtn" onclick="sendQuery()" disabled>↑</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let fileSelected = null;
let docReady = false;
let pollTimer = null;

document.getElementById("fileInput").addEventListener("change", function(e) {
  if (e.target.files[0]) selectFile(e.target.files[0]);
});

const dropZone = document.getElementById("dropZone");
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag");
  const f = e.dataTransfer.files[0];
  if (f && f.type === "application/pdf") selectFile(f);
  else showToast("Please drop a PDF file.", "error");
});

function selectFile(f) {
  fileSelected = f;
  document.getElementById("uploadBtn").disabled = false;
  document.getElementById("fileName").textContent = f.name;
  document.getElementById("fileInfo").classList.add("show");
  document.getElementById("chunkCount").textContent = "—";
  docReady = false;
  disableChat();
}

async function uploadPDF() {
  if (!fileSelected) return;
  document.getElementById("uploadBtn").disabled = true;
  document.getElementById("progressWrap").classList.add("show");
  document.getElementById("progressMsg").textContent = "Uploading file...";

  const fd = new FormData();
  fd.append("pdf", fileSelected);

  try {
    const res = await fetch("/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (data.success) {
      document.getElementById("statusPill").textContent = "⟳ Indexing...";
      document.getElementById("statusPill").className = "status-pill processing";
      startPolling();
    } else {
      document.getElementById("progressWrap").classList.remove("show");
      document.getElementById("uploadBtn").disabled = false;
      showToast(data.error || "Upload failed.", "error");
    }
  } catch(e) {
    document.getElementById("progressWrap").classList.remove("show");
    document.getElementById("uploadBtn").disabled = false;
    showToast("Upload failed: " + e.message, "error");
  }
}

function startPolling() {
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch("/status");
      const data = await res.json();
      document.getElementById("progressMsg").textContent = data.progress_msg || "Processing...";

      if (data.status === "ready") {
        clearInterval(pollTimer);
        const pb = document.getElementById("progressBar");
        pb.classList.remove("running");
        pb.classList.add("done");
        document.getElementById("chunkCount").textContent = data.num_chunks;
        document.getElementById("statusPill").textContent = "● " + data.loaded_file;
        document.getElementById("statusPill").className = "status-pill ready";
        document.getElementById("chatHeader").textContent = "Querying: " + data.loaded_file;
        docReady = true;
        enableChat();
        showToast("Document indexed — ready to query!", "success");
        setTimeout(() => document.getElementById("progressWrap").classList.remove("show"), 2500);

      } else if (data.status === "error") {
        clearInterval(pollTimer);
        document.getElementById("progressWrap").classList.remove("show");
        document.getElementById("uploadBtn").disabled = false;
        document.getElementById("statusPill").textContent = "No document loaded";
        document.getElementById("statusPill").className = "status-pill";
        showToast("Error: " + data.error, "error");
      }
    } catch(e) {
      // Transient network blip — keep polling
    }
  }, 2000);
}

async function sendQuery() {
  const input = document.getElementById("queryInput");
  const q = input.value.trim();
  if (!q || !docReady) return;
  input.value = "";
  autoResize(input);
  appendMsg(q, "user");
  const thinkId = appendThinking();
  document.getElementById("sendBtn").disabled = true;
  input.disabled = true;

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q })
    });
    const data = await res.json();
    removeMsg(thinkId);
    appendMsg(data.answer || ("⚠ " + (data.error || "No answer returned.")), "bot");
  } catch(e) {
    removeMsg(thinkId);
    appendMsg("⚠ Could not reach the server.", "bot");
  } finally {
    document.getElementById("sendBtn").disabled = false;
    input.disabled = false;
    input.focus();
  }
}

let msgId = 0;
function appendMsg(text, who) {
  document.getElementById("emptyState")?.remove();
  const id = "msg-" + (++msgId);
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.id = id;
  div.innerHTML = `<div class="msg-avatar">${who === "user" ? "👤" : "🤖"}</div><div class="msg-body">${esc(text)}</div>`;
  document.getElementById("messages").appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
  return id;
}

function appendThinking() {
  document.getElementById("emptyState")?.remove();
  const id = "msg-" + (++msgId);
  const div = document.createElement("div");
  div.className = "msg bot"; div.id = id;
  div.innerHTML = `<div class="msg-avatar">🤖</div><div class="msg-body thinking">Thinking <div class="dots"><span></span><span></span><span></span></div></div>`;
  document.getElementById("messages").appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
  return id;
}

function removeMsg(id) { document.getElementById(id)?.remove(); }
function esc(s) { return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function useSuggestion(btn) {
  if (!docReady) { showToast("Please load a document first.", "error"); return; }
  document.getElementById("queryInput").value = btn.textContent;
  sendQuery();
}
function enableChat()  { document.getElementById("queryInput").disabled = false; document.getElementById("sendBtn").disabled = false; }
function disableChat() { document.getElementById("queryInput").disabled = true;  document.getElementById("sendBtn").disabled = true; }
function handleKey(e)  { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuery(); } }
function autoResize(el) { el.style.height = "auto"; el.style.height = Math.min(el.scrollHeight, 140) + "px"; }
function showToast(msg, type="success") {
  const t = document.getElementById("toast");
  t.textContent = msg; t.className = "toast " + type + " show";
  setTimeout(() => t.className = "toast", 4000);
}
</script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/upload', methods=['POST'])
def upload():
    """Save the PDF and immediately kick off background indexing."""
    if 'pdf' not in request.files:
        return jsonify({"success": False, "error": "No file provided."})
    f = request.files['pdf']
    if not f.filename.lower().endswith('.pdf'):
        return jsonify({"success": False, "error": "Only PDF files are accepted."})

    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)

    
    rag_state.update({
        "db": None, "rag_chain": None, "loaded_file": None,
        "num_chunks": 0, "status": "processing",
        "progress_msg": "Starting...", "error": ""
    })

    
    threading.Thread(target=init_rag_background, args=(path, filename), daemon=True).start()
    return jsonify({"success": True})


@app.route('/status')
def status():
    """Polled every 2 s by the browser to track indexing progress."""
    return jsonify({
        "status":       rag_state["status"],
        "progress_msg": rag_state["progress_msg"],
        "loaded_file":  rag_state["loaded_file"],
        "num_chunks":   rag_state["num_chunks"],
        "error":        rag_state["error"],
    })


@app.route('/query', methods=['POST'])
def query():
    if rag_state["status"] != "ready":
        return jsonify({"error": "Document not ready yet."})
    data = request.json
    q = (data.get('query') or '').strip()
    if not q:
        return jsonify({"error": "Empty query."})
    try:
        answer = rag_state["rag_chain"].invoke(q)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    
    threading.Thread(target=get_embeddings, daemon=True).start()
    app.run(debug=True, port=5000, threaded=True)
