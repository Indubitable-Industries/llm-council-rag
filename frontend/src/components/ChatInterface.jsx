import { useState, useEffect, useRef, memo } from 'react';
import { useDropzone } from 'react-dropzone';
import ReactMarkdown from 'react-markdown';
import { api } from '../api';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import './ChatInterface.css';

function RepoDropzone({ conversationId, onIndexed }) {
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const progressTimer = useRef(null);

  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.zip')) {
      setStatus('Please drop a .zip file.');
      return;
    }

    setStatus('Uploading…');
    setIsUploading(true);
    setProgress(5);

    if (progressTimer.current) clearInterval(progressTimer.current);
    progressTimer.current = setInterval(() => {
      setProgress((p) => {
        if (p >= 90) return 90;
        return p + 5;
      });
    }, 200);

    try {
      const data = await api.uploadRepo(conversationId, file);
      if (data.status === 'success') {
        setStatus('Indexing…');
        // Show an indexing spinner state until the response returns.
        setProgress(90);
        setStatus(data.message || 'Repository indexed successfully.');
        setProgress(100);
        onIndexed && onIndexed();
      } else {
        setStatus(data.message || 'Indexing failed.');
        setProgress(100);
      }
    } catch (err) {
      console.error(err);
      setStatus(err.message || 'Upload failed. Check backend logs.');
      setProgress(0);
    }

    if (progressTimer.current) {
      clearInterval(progressTimer.current);
      progressTimer.current = null;
    }
    setIsUploading(false);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  useEffect(() => {
    return () => {
      if (progressTimer.current) clearInterval(progressTimer.current);
    };
  }, []);

  return (
    <div className="repo-dropzone-wrapper">
      <div
        {...getRootProps()}
        className="repo-dropzone"
        style={{
          border: '2px dashed #555',
          borderRadius: '0.5rem',
          padding: '0.75rem',
          fontSize: '0.85rem',
          cursor: 'pointer',
          marginBottom: '0.75rem',
          backgroundColor: isDragActive ? '#1f1f1f' : 'transparent',
        }}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <span>Drop your repo .zip here…</span>
        ) : (
          <span>
            Drag &amp; drop a repo <code>.zip</code> here to give the council local
            code context for this conversation.
          </span>
        )}
      </div>
      {status && (
        <div
          style={{
            fontSize: '0.8rem',
            marginTop: '0.25rem',
            opacity: 0.9,
          }}
        >
          {status}
        </div>
      )}
      {(isUploading || progress > 0) && (
        <div
          style={{
            marginTop: '0.35rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <div
            style={{
              flex: 1,
              background: '#2a2a2a',
              borderRadius: '0.35rem',
              overflow: 'hidden',
              height: '6px',
            }}
          >
            <div
              style={{
                width: `${progress}%`,
                transition: 'width 0.2s ease',
                background: '#4caf50',
                height: '100%',
              }}
            />
          </div>
          {progress >= 90 && progress < 100 && (
            <div className="tiny-spinner" title="Indexing" />
          )}
        </div>
      )}
    </div>
  );
}

export default function ChatInterface({
  conversation,
  onSendMessage,
  onStop,
  isLoading,
}) {
  const [input, setInput] = useState('');
  const [manualSelections, setManualSelections] = useState([]);
  const [repoTree, setRepoTree] = useState([]);
  const [contextPanelOpen, setContextPanelOpen] = useState(false);
  const [resolvingDirectives, setResolvingDirectives] = useState(false);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const [showScrollDown, setShowScrollDown] = useState(false);

  const refreshRepoTree = () => {
    if (!conversation?.id) return;
    api
      .getRepoTree(conversation.id)
      .then((tree) => setRepoTree(tree))
      .catch((err) => console.error('Failed to load repo tree', err));
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  useEffect(() => {
    const el = messagesContainerRef.current;
    if (!el) return;

    const handleScroll = () => {
      const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      setShowScrollDown(distanceFromBottom > 200);
    };

    el.addEventListener('scroll', handleScroll);
    handleScroll();
    return () => el.removeEventListener('scroll', handleScroll);
  }, [conversation?.id]);

  const parseDirectives = (text) => {
    const regex = /@("[^"]+"|\S+)/g;
    const directives = [];
    let cleaned = text;
    let match;
    while ((match = regex.exec(text)) !== null) {
      directives.push(match[0]);
      cleaned = cleaned.replace(match[0], '').trim();
    }
    return { directives, cleaned };
  };

  const resolveDirectives = async (directives, userText) => {
    const resolved = [];
    for (const dir of directives) {
      let token = dir.startsWith('@') ? dir.slice(1) : dir;
      if (token.startsWith('file:')) {
        token = token.replace('file:', '');
      }

      try {
        const pathRes = await api.resolvePath(conversation.id, token, userText);
        const match = pathRes?.matches?.[0];
        if (match?.content) {
          resolved.push({
            path: match.path,
            content: match.content,
            source_type: 'manual_at',
            score: match.score,
          });
          continue;
        }
      } catch (err) {
        console.error('Path resolve failed', err);
      }

      try {
        const searchRes = await api.searchRepo(conversation.id, token, 1);
        const result = searchRes?.results?.[0];
        if (result?.snippet) {
          resolved.push({
            path: result.path,
            content: result.snippet,
            source_type: 'manual_at_snippet',
          });
        }
      } catch (err) {
        console.error('Search failed', err);
      }
    }
    return resolved;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading || resolvingDirectives) return;

    try {
      setResolvingDirectives(true);
      const { directives, cleaned } = parseDirectives(input);
      const directiveContexts = await resolveDirectives(directives, cleaned);

      const manualContext = [
        ...manualSelections,
        ...directiveContexts,
      ];

      await onSendMessage(cleaned, manualContext);
      setInput('');
      // Clear manual selections after a send so the picker resets.
      setManualSelections([]);
    } catch (err) {
      console.error('Failed to send message', err);
    } finally {
      setResolvingDirectives(false);
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  useEffect(() => {
    if (!conversation?.id) {
      setRepoTree([]);
      setManualSelections([]);
      return;
    }

    // Reset picker state on conversation change so prior repos don't linger visually.
    setRepoTree([]);
    setManualSelections([]);
    refreshRepoTree();
  }, [conversation?.id]);

  const handleAddFileContext = async (path) => {
    try {
      const file = await api.getFile(conversation.id, path);
      setManualSelections((prev) => {
        if (prev.find((p) => p.path === path)) return prev;
        return [...prev, { path, content: file.content, source_type: 'manual_picker' }];
      });
    } catch (err) {
      console.error('Failed to add file', err);
    }
  };

  const handleRemoveManual = (path) => {
    setManualSelections((prev) => prev.filter((p) => p.path !== path));
  };

  const renderRepoTree = (nodes) => {
    if (!nodes || nodes.length === 0) return <div className="repo-tree-empty">No repo uploaded.</div>;

    return nodes.map((node) => {
      if (node.type === 'dir') {
        return (
          <details key={node.path} open>
            <summary>{node.name}</summary>
            <div className="repo-tree-children">{renderRepoTree(node.children)}</div>
          </details>
        );
      }
      return (
        <div
          key={node.path}
          className="repo-tree-file"
          onClick={() => handleAddFileContext(node.path)}
          role="button"
        >
          {node.name}
        </div>
      );
    });
  };

  const ContextPanel = memo(({ sources }) => {
    if (!sources || sources.length === 0) return null;
    const manual = sources.filter((s) => (s.source_type || '').startsWith('manual'));
    const rag = sources.filter((s) => (s.source_type || '').startsWith('rag'));

    const uniqueFiles = new Set((sources || []).map((s) => s.source)).size;
    const lineTotal = (sources || []).reduce((sum, s) => sum + (s.lines || 0), 0);

    const [collapsed, setCollapsed] = useState(true);

    const renderSection = (label, items) => (
      <div className="context-section">
        <div className="context-section-title">{label}</div>
        {items.map((item, idx) => {
          const isFullFile = (item.source_type || '').startsWith('manual') && item.content && item.content.length > 0;
          const showPreview = !isFullFile || (item.source_type || '').includes('snippet') || (item.source_type || '').startsWith('rag');
          return (
            <div className="context-row" key={`${item.source}-${idx}`}>
              <div className="context-row-header">
                <span className="context-tag">{label === 'Manual' ? 'Manual' : 'RAG'}</span>
                <span className="context-path">{item.source}</span>
                {item.score !== null && item.score !== undefined && (
                  <span className="context-score">score {item.score.toFixed(3)}</span>
                )}
                <span className="context-meta">{item.lines} lines · {item.est_tokens || 0} est tokens</span>
              </div>
              {showPreview ? (
                <div className="context-preview">{item.content || ''}</div>
              ) : (
                <div className="context-preview muted">(Full file included; content hidden)</div>
              )}
            </div>
          );
        })}
      </div>
    );

    return (
      <div className="context-panel">
        <div className="context-header" onClick={() => setCollapsed((v) => !v)} role="button">
          <div className="context-title">Context used · {uniqueFiles} files · {lineTotal} lines</div>
          <div className="context-toggle-indicator">{collapsed ? 'Show' : 'Hide'}</div>
        </div>
        {!collapsed && (
          <>
            {manual.length > 0 && renderSection('Manual', manual)}
            {rag.length > 0 && renderSection('RAG', rag)}
          </>
        )}
      </div>
    );
  }, (prev, next) => prev.sources === next.sources);

  if (!conversation) {
    return (
      <div className="chat-interface">
        <div className="empty-state">
          <h2>Welcome to LLM Council</h2>
          <p>Create a new conversation to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-interface">
      {/* New: repo dropzone for this conversation */}
      <RepoDropzone conversationId={conversation.id} onIndexed={refreshRepoTree} />

      <div className="context-tools">
        <div className="context-tools-header">
          <div>
            <div className="context-tools-title">Manual context</div>
            <div className="context-tools-subtitle">
              Click files to add, or use @file:path / @token in your message. Manual context skips auto-RAG.
            </div>
          </div>
          <button className="context-toggle" onClick={() => setContextPanelOpen((v) => !v)}>
            {contextPanelOpen ? 'Hide picker' : 'Show picker'}
          </button>
        </div>
        {contextPanelOpen && (
          <div className="context-picker">
            <div className="repo-tree" aria-label="Repository tree">
              {renderRepoTree(repoTree)}
            </div>
            <div className="selected-context">
              <div className="selected-context-title">Selected for next message</div>
              {manualSelections.length === 0 && <div className="selected-context-empty">None yet</div>}
              {manualSelections.map((item) => (
                <div className="selected-chip" key={item.path}>
                  <span className="chip-path">{item.path}</span>
                  <button className="chip-remove" onClick={() => handleRemoveManual(item.path)}>×</button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="messages-container" ref={messagesContainerRef}>
        {conversation.messages.length === 0 ? (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>Ask a question to consult the LLM Council</p>
          </div>
        ) : (
          conversation.messages.map((msg, index) => (
            <div key={index} className="message-group">
              {msg.role === 'user' ? (
                <div className="user-message">
                  <div className="message-label">You</div>
                  <div className="message-content">
                    <div className="markdown-content">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="assistant-message">
                  <div className="message-label">LLM Council</div>

                  {/* Stage 1 */}
                  {msg.loading?.stage1 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>
                        Running Stage 1: Collecting individual responses...
                      </span>
                    </div>
                  )}
                  {msg.stage1 && <Stage1 responses={msg.stage1} />}

                  {/* Stage 2 */}
                  {msg.loading?.stage2 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 2: Peer rankings...</span>
                    </div>
                  )}
                  {msg.stage2 && (
                    <Stage2
                      rankings={msg.stage2}
                      labelToModel={msg.metadata?.label_to_model}
                      aggregateRankings={msg.metadata?.aggregate_rankings}
                    />
                  )}

                  {/* Stage 3 */}
                  {msg.loading?.stage3 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 3: Final synthesis...</span>
                    </div>
                  )}
                  {msg.stage3 && <Stage3 finalResponse={msg.stage3} />}

                  <ContextPanel sources={msg.contextSources || msg.context_sources} />
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <span>Consulting the council...</span>
            {onStop && (
              <button className="stop-button" type="button" onClick={onStop}>
                Stop
              </button>
            )}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {showScrollDown && (
        <button className="scroll-bottom" type="button" onClick={scrollToBottom}>
          ↓
        </button>
      )}

      <form className="input-form" onSubmit={handleSubmit}>
        <textarea
          className="message-input"
          placeholder="Ask your question... (Shift+Enter for new line, Enter to send)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading || resolvingDirectives}
          rows={3}
        />
        <button
          type="submit"
          className="send-button"
          disabled={!input.trim() || isLoading || resolvingDirectives}
        >
          {resolvingDirectives ? 'Resolving…' : 'Send'}
        </button>
      </form>
    </div>
  );
}
