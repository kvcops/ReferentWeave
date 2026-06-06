document.addEventListener("DOMContentLoaded", () => {
    // Icons initialization
    lucide.createIcons();

    // App state
    let activeFilterDoc = null;

    // Elements
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabs = document.querySelectorAll(".tab-content");
    const dragDropArea = document.getElementById("drag-drop-area");
    const fileInput = document.getElementById("file-input");
    const progressContainer = document.getElementById("upload-progress-container");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const progressFilename = document.getElementById("progress-filename");
    const progressPercent = document.getElementById("progress-percent");
    const consoleLogs = document.getElementById("console-logs");
    const clearLogsBtn = document.getElementById("clear-logs");
    const docsList = document.getElementById("docs-list");
    const chatForm = document.getElementById("chat-form");
    const queryInput = document.getElementById("query-input");
    const chatMessages = document.getElementById("chat-messages");
    const queryTraceConsole = document.getElementById("query-trace-console");
    const chunksContainer = document.getElementById("chunks-container");
    const filterIndicator = document.getElementById("filter-indicator");
    const activeFilterDocSpan = document.getElementById("active-filter-doc");
    const removeFilterBtn = document.getElementById("remove-filter-btn");

    // Tab buttons for right side panel in Chat Studio
    const traceTabButtons = document.querySelectorAll(".trace-tab-btn");
    const traceTabContents = document.querySelectorAll(".trace-tab-content");

    // Tab Navigation
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            tabs.forEach(t => t.classList.remove("active"));

            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
        });
    });

    // Chat Studio subtabs switching (Pipeline Trace vs Source Chunks)
    traceTabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            traceTabButtons.forEach(b => b.classList.remove("active"));
            traceTabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const subTabId = btn.getAttribute("data-trace-tab");
            document.getElementById(subTabId).classList.add("active");
        });
    });

    // Logging Console helper
    function appendLog(text, type = "info") {
        const line = document.createElement("div");
        line.className = `console-line ${type}`;
        line.innerText = text;
        consoleLogs.appendChild(line);
        consoleLogs.scrollTop = consoleLogs.scrollHeight;
    }

    clearLogsBtn.addEventListener("click", () => {
        consoleLogs.innerHTML = "";
        appendLog("Console cleared.", "system");
    });

    // Fetch and render document list
    async function loadDocuments() {
        try {
            const res = await fetch("/api/documents");
            const data = await res.json();
            
            if (!data.documents || data.documents.length === 0) {
                docsList.innerHTML = `
                    <div class="empty-docs-state">
                        No documents uploaded yet.
                    </div>
                `;
                return;
            }

            docsList.innerHTML = "";
            data.documents.forEach(doc => {
                const item = document.createElement("div");
                item.className = "doc-item";
                if (activeFilterDoc === doc) {
                    item.classList.add("selected");
                }

                item.innerHTML = `
                    <div class="doc-name" title="${doc}">
                        <i data-lucide="file-text"></i>
                        <span>${doc}</span>
                    </div>
                    <div class="doc-actions">
                        <button class="doc-del-btn" data-doc="${doc}">
                            <i data-lucide="trash-2"></i>
                        </button>
                    </div>
                `;

                // Set file filter on click
                item.addEventListener("click", (e) => {
                    // Prevent trigger if trash was clicked
                    if (e.target.closest(".doc-del-btn")) return;

                    if (activeFilterDoc === doc) {
                        activeFilterDoc = null;
                        item.classList.remove("selected");
                        filterIndicator.classList.add("hidden");
                    } else {
                        activeFilterDoc = doc;
                        document.querySelectorAll(".doc-item").forEach(d => d.classList.remove("selected"));
                        item.classList.add("selected");
                        activeFilterDocSpan.innerText = doc;
                        filterIndicator.classList.remove("hidden");
                    }
                });

                docsList.appendChild(item);
            });
            lucide.createIcons();

            // Re-bind delete button events
            document.querySelectorAll(".doc-del-btn").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const docId = btn.getAttribute("data-doc");
                    if (confirm(`Are you sure you want to delete the document '${docId}'?`)) {
                        await deleteDocument(docId);
                    }
                });
            });

        } catch (e) {
            console.error("Failed to load documents", e);
        }
    }

    // Delete document
    async function deleteDocument(docId) {
        try {
            const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
            const data = await res.json();
            if (res.ok) {
                appendLog(`Deleted document: ${docId}`, "system");
                if (activeFilterDoc === docId) {
                    activeFilterDoc = null;
                    filterIndicator.classList.add("hidden");
                }
                loadDocuments();
            } else {
                appendLog(`Error deleting document: ${data.detail}`, "error");
            }
        } catch (e) {
            appendLog(`Connection error during delete: ${e.message}`, "error");
        }
    }

    // Upload functionality
    dragDropArea.addEventListener("click", () => fileInput.click());

    dragDropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dragDropArea.classList.add("dragover");
    });

    dragDropArea.addEventListener("dragleave", () => {
        dragDropArea.classList.remove("dragover");
    });

    dragDropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        dragDropArea.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleUpload(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleUpload(e.target.files[0]);
        }
    });

    async function handleUpload(file) {
        if (!file.name.endsWith(".pdf")) {
            appendLog(`Rejected upload of '${file.name}': Only PDF files are supported.`, "error");
            return;
        }

        progressContainer.classList.remove("hidden");
        progressFilename.innerText = file.name;
        progressBarFill.style.width = "0%";
        progressPercent.innerText = "0%";
        appendLog(`Starting upload sequence for '${file.name}'...`, "system");

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Simulated upload progress since standard fetch doesn't support progress events easily.
            // We just animate it to 85% and complete on response.
            let prog = 0;
            const progInterval = setInterval(() => {
                if (prog < 85) {
                    prog += 5;
                    progressBarFill.style.width = `${prog}%`;
                    progressPercent.innerText = `${prog}%`;
                }
            }, 100);

            const response = await fetch("/api/ingest", {
                method: "POST",
                body: formData
            });

            clearInterval(progInterval);

            const result = await response.json();
            
            if (response.ok) {
                progressBarFill.style.width = "100%";
                progressPercent.innerText = "100%";
                
                appendLog(`Upload sequence complete: ${result.filename}`, "success");
                appendLog(`Created ${result.chunks_count} chunks. Resolved ${result.resolutions_count} references.`, "success");
                
                // Print detailed trace logs from backend pipeline
                if (result.trace) {
                    result.trace.forEach(log => {
                        let logType = "info";
                        if (log.toLowerCase().includes("fail") || log.toLowerCase().includes("error")) {
                            logType = "error";
                        } else if (log.toLowerCase().includes("resolved")) {
                            logType = "success";
                        } else if (log.toLowerCase().includes("starting") || log.toLowerCase().includes("generating")) {
                            logType = "system";
                        }
                        appendLog(log, logType);
                    });
                }
                
                // Reload sidebar doc list
                loadDocuments();
                
                // Reset progress UI after delay
                setTimeout(() => {
                    progressContainer.classList.add("hidden");
                }, 3000);
            } else {
                progressBarFill.style.width = "0%";
                progressPercent.innerText = "Error";
                appendLog(`Ingestion failed: ${result.detail}`, "error");
            }
        } catch (e) {
            progressBarFill.style.width = "0%";
            progressPercent.innerText = "Error";
            appendLog(`Network connection error during ingestion: ${e.message}`, "error");
        }
    }

    // Filter indicator controls
    removeFilterBtn.addEventListener("click", () => {
        activeFilterDoc = null;
        document.querySelectorAll(".doc-item").forEach(d => d.classList.remove("selected"));
        filterIndicator.classList.add("hidden");
    });

    // Chat / Query submission
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // Reset inputs
        queryInput.value = "";
        
        // 1. Render User Message
        appendMessage(query, "user");

        // 2. Render Assistant Loading skeleton
        const loadingMsgId = appendMessage('<div class="skeleton-dots">Thinking<span>.</span><span>.</span><span>.</span></div>', "assistant");

        // 3. Clear Trace Console & Chunks
        queryTraceConsole.innerHTML = "";
        chunksContainer.innerHTML = "";

        // Prepare request body
        const reqBody = {
            query: query
        };
        if (activeFilterDoc) {
            reqBody.doc_ids = [activeFilterDoc];
        }

        try {
            const res = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(reqBody)
            });

            const data = await res.json();
            
            // Remove loading msg
            document.getElementById(loadingMsgId).remove();

            if (res.ok) {
                // 4. Render Grounded Answer (parse markdown and citations into styled HTML)
                const parsedMarkdown = parseMarkdown(data.answer);
                const answerHTML = formatAnswerCitations(parsedMarkdown);
                appendMessage(answerHTML, "assistant");

                // 5. Render Pipeline Trace Logs
                if (data.trace) {
                    queryTraceConsole.innerHTML = "";
                    data.trace.forEach(log => {
                        const line = document.createElement("div");
                        line.className = "console-line";
                        line.innerText = log;
                        queryTraceConsole.appendChild(line);
                    });
                }

                // 6. Render Grounding Chunks
                if (data.candidates && data.candidates.length > 0) {
                    chunksContainer.innerHTML = "";
                    data.candidates.forEach((chunk, index) => {
                        const card = document.createElement("div");
                        card.className = "chunk-card";
                        card.id = `chunk-card-${chunk.chunk_id}`;

                        // Extract resolution items for rendering
                        let resolvedHTML = "";
                        if (chunk.enriched_text.includes("[Resolved Context Block:")) {
                            const match = chunk.enriched_text.match(/\[Resolved Context Block:\n([\s\S]+?)\]/);
                            if (match) {
                                const lines = match[1].trim().split("\n");
                                resolvedHTML = `
                                    <div class="chunk-resolved-block">
                                        <div class="chunk-block-title">Cheat Sheet Appended</div>
                                        ${lines.map(l => `<div class="chunk-resolved-line">${l}</div>`).join("")}
                                    </div>
                                `;
                            }
                        }

                        card.innerHTML = `
                            <div class="chunk-header">
                                <span class="chunk-title"><i data-lucide="database"></i> Chunk ID: ${chunk.chunk_id}</span>
                                <span class="chunk-score-badge">Score: ${chunk.score.toFixed(3)}</span>
                            </div>
                            <div class="chunk-body">
                                <div class="chunk-block-title">Untouched Source Text</div>
                                <div>${chunk.text}</div>
                                ${resolvedHTML}
                            </div>
                            <div class="chunk-meta-info">
                                Source Document: ${chunk.doc_id}
                            </div>
                        `;
                        chunksContainer.appendChild(card);
                    });
                    lucide.createIcons();
                    
                    // Bind citation click triggers
                    bindCitations();
                } else {
                    chunksContainer.innerHTML = `
                        <div class="empty-trace-state">
                            <i data-lucide="database"></i>
                            <p>No grounding chunk documents utilized.</p>
                        </div>
                    `;
                }

            } else {
                appendMessage(`Error: ${data.detail}`, "assistant");
            }

        } catch (e) {
            document.getElementById(loadingMsgId).remove();
            appendMessage(`Network connection failed: ${e.message}`, "assistant");
        }
    });

    // Helper: Add message bubble
    function appendMessage(content, sender) {
        const id = `msg-${Date.now()}`;
        const bubble = document.createElement("div");
        bubble.className = `message ${sender}`;
        bubble.id = id;

        bubble.innerHTML = `
            <div class="message-content">
                ${content}
            </div>
        `;
        
        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return id;
    }

    // Helper: Find brackets like [1001] and make them clickable
    function formatAnswerCitations(answer) {
        // Regex to match [1000] - [1050]
        const citationRegex = /\[(\d{4})\]/g;
        return answer.replace(citationRegex, (match, chunkId) => {
            return `<button class="citation-link" data-chunk-id="${chunkId}">[${chunkId}]</button>`;
        });
    }

    // Helper: Parse Markdown syntax (headers, lists, bold, italic, links, code) into styled HTML
    function parseMarkdown(text) {
        if (!text) return "";

        const lines = text.split("\n");
        let htmlLines = [];
        let inList = false;
        let listType = null; // 'ul' or 'ol'
        let inCodeBlock = false;

        // Helper to format inline elements: bold, italic, links, inline code
        function formatInline(txt) {
            // Escape HTML tags to prevent XSS and rendering issues, but keep our markdown parsers
            let escaped = txt
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;");

            // Bold: **text** or __text__
            escaped = escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
            escaped = escaped.replace(/__([^_]+)__/g, "<strong>$1</strong>");

            // Italic: *text* or _text_
            escaped = escaped.replace(/\*([^*]+)\*/g, "<em>$1</em>");
            escaped = escaped.replace(/_([^_]+)_/g, "<em>$1</em>");

            // Inline code: `code`
            escaped = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");

            // Links: [text](url)
            escaped = escaped.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

            return escaped;
        }

        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];
            
            // Handle code blocks
            if (line.trim().startsWith("```")) {
                if (inCodeBlock) {
                    htmlLines.push("</code></pre>");
                    inCodeBlock = false;
                } else {
                    htmlLines.push("<pre><code>");
                    inCodeBlock = true;
                }
                continue;
            }

            if (inCodeBlock) {
                // Inside code block, escape and keep raw line
                let escapedLine = line
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;");
                htmlLines.push(escapedLine);
                continue;
            }

            let trimmed = line.trim();

            // Handle empty lines
            if (trimmed === "") {
                if (inList) {
                    htmlLines.push(`</${listType}>`);
                    inList = false;
                    listType = null;
                }
                htmlLines.push("<div class='chat-space'></div>");
                continue;
            }

            // Headers
            const headerMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
            if (headerMatch) {
                if (inList) {
                    htmlLines.push(`</${listType}>`);
                    inList = false;
                    listType = null;
                }
                const level = headerMatch[1].length;
                const content = formatInline(headerMatch[2]);
                htmlLines.push(`<h${level}>${content}</h${level}>`);
                continue;
            }

            // Lists
            const bulletMatch = trimmed.match(/^[\*\-]\s+(.*)$/);
            const numberMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);

            if (bulletMatch) {
                if (!inList || listType !== "ul") {
                    if (inList) htmlLines.push(`</${listType}>`);
                    htmlLines.push("<ul>");
                    inList = true;
                    listType = "ul";
                }
                const content = formatInline(bulletMatch[1]);
                htmlLines.push(`<li>${content}</li>`);
                continue;
            }

            if (numberMatch) {
                if (!inList || listType !== "ol") {
                    if (inList) htmlLines.push(`</${listType}>`);
                    htmlLines.push("<ol>");
                    inList = true;
                    listType = "ol";
                }
                const content = formatInline(numberMatch[2]);
                htmlLines.push(`<li>${content}</li>`);
                continue;
            }

            // Standard paragraph
            if (inList) {
                htmlLines.push(`</${listType}>`);
                inList = false;
                listType = null;
            }

            const content = formatInline(trimmed);
            htmlLines.push(`<p>${content}</p>`);
        }

        if (inList) {
            htmlLines.push(`</${listType}>`);
        }
        if (inCodeBlock) {
            htmlLines.push("</code></pre>");
        }

        return htmlLines.join("\n");
    }

    // Helper: Bind citation button click scroll handlers
    function bindCitations() {
        document.querySelectorAll(".citation-link").forEach(btn => {
            btn.addEventListener("click", () => {
                const chunkId = btn.getAttribute("data-chunk-id");
                const targetCard = document.getElementById(`chunk-card-${chunkId}`);
                
                if (targetCard) {
                    // Activate Chunks subtab
                    document.querySelector("[data-trace-tab='source-chunks']").click();
                    
                    // Highlight target chunk card
                    document.querySelectorAll(".chunk-card").forEach(c => c.classList.remove("highlighted"));
                    targetCard.classList.add("highlighted");
                    
                    // Scroll to card
                    targetCard.scrollIntoView({ behavior: "smooth", block: "center" });
                } else {
                    alert(`Chunk ID ${chunkId} was not retrieved in the top reranked chunks.`);
                }
            });
        });
    }

    // Load documents on boot
    loadDocuments();
});
