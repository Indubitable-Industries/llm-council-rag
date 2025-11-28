import { useState, useEffect } from 'react';
import { api } from '../api';
import './Sidebar.css';

export default function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
}) {
  const [mode, setMode] = useState('baseline');
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({ council_models: [], chairman_model: '' });
  const [settingsStatus, setSettingsStatus] = useState('');

  useEffect(() => {
    if (showSettings) {
      api.getSettings()
        .then((data) => setSettings(data))
        .catch(() => setSettingsStatus('Failed to load settings'));
    }
  }, [showSettings]);

  const handleNew = () => {
    onNewConversation(mode);
  };

  const handleSettingsSave = async () => {
    try {
      setSettingsStatus('Saving...');
      const payload = {
        council_models: settings.council_models,
        chairman_model: settings.chairman_model,
      };
      const saved = await api.updateSettings(payload);
      setSettings(saved);
      setSettingsStatus('Saved');
    } catch (err) {
      setSettingsStatus('Save failed');
    }
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>LLM Council</h1>
        <div className="new-convo-controls">
          <select
            className="mode-select"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            <option value="baseline">Baseline</option>
            <option value="round_robin">Round Robin</option>
            <option value="fight">Fight</option>
            <option value="stacks">Stacks</option>
            <option value="complex_iterative">Complex Iterative</option>
            <option value="complex_questioning">Complex Questioning</option>
          </select>
          <button className="new-conversation-btn" onClick={handleNew}>
          + New Conversation
        </button>
        </div>
        <button
          className="settings-toggle"
          onClick={() => setShowSettings((v) => !v)}
        >
          {showSettings ? 'Hide settings' : 'Show settings'}
        </button>
      </div>

      {showSettings && (
        <div className="settings-panel">
          <div className="settings-field">
            <label>Council models (one per line)</label>
            <textarea
              value={(settings.council_models || []).join('\n')}
              onChange={(e) =>
                setSettings((prev) => ({
                  ...prev,
                  council_models: e.target.value
                    .split('\n')
                    .map((v) => v.trim())
                    .filter(Boolean),
                }))
              }
            />
          </div>
          <div className="settings-field">
            <label>Chairman model</label>
            <input
              type="text"
              value={settings.chairman_model || ''}
              onChange={(e) =>
                setSettings((prev) => ({ ...prev, chairman_model: e.target.value }))
              }
            />
          </div>
          <div className="settings-actions">
            <button className="new-conversation-btn" onClick={handleSettingsSave}>
              Save settings
            </button>
            {settingsStatus && <span className="settings-status">{settingsStatus}</span>}
          </div>
        </div>
      )}

      <div className="conversation-list">
        {conversations.length === 0 ? (
          <div className="no-conversations">No conversations yet</div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${
                conv.id === currentConversationId ? 'active' : ''
              }`}
              onClick={() => onSelectConversation(conv.id)}
            >
              <div className="conversation-title">
                {conv.title || 'New Conversation'}
              </div>
              <div className="conversation-meta">
                {conv.message_count} messages · {conv.mode || 'baseline'}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
