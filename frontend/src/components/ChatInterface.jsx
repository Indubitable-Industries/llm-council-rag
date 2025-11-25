import { useState, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import ReactMarkdown from 'react-markdown';
import { api } from '../api';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import './ChatInterface.css';

function RepoDropzone({ conversationId }) {
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

    setStatus('Uploading and indexing repo…');
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
        setStatus(data.message || 'Repository indexed successfully.');
      } else {
        setStatus(data.message || 'Indexing failed.');
      }
      setProgress(100);
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
      )}
    </div>
  );
}

export default function ChatInterface({
  conversation,
  onSendMessage,
  isLoading,
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

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
      <RepoDropzone conversationId={conversation.id} />

      <div className="messages-container">
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

                  {msg.ragContextSources?.length > 0 && (
                    <div
                      style={{
                        marginTop: '0.5rem',
                        padding: '0.5rem',
                        border: '1px solid #333',
                        borderRadius: '0.35rem',
                        fontSize: '0.85rem',
                        background: '#151515',
                      }}
                    >
                      <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                        Context used:
                      </div>
                      <ul style={{ paddingLeft: '1.1rem', margin: 0 }}>
                        {msg.ragContextSources.map((src, i) => (
                          <li key={i} style={{ marginBottom: '0.15rem' }}>
                            <code>{src.source}</code>
                            {src.lines ? ` — ${src.lines} lines` : ''}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <span>Consulting the council...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {conversation.messages.length === 0 && (
        <form className="input-form" onSubmit={handleSubmit}>
          <textarea
            className="message-input"
            placeholder="Ask your question... (Shift+Enter for new line, Enter to send)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            rows={3}
          />
          <button
            type="submit"
            className="send-button"
            disabled={!input.trim() || isLoading}
          >
            Send
          </button>
        </form>
      )}
    </div>
  );
}
