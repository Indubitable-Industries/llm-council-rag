/**
 * API client for the LLM Council backend.
 */

// Allow overriding the backend URL via Vite env; default to local dev backend.
export const API_BASE =
  import.meta.env.VITE_API_BASE || 'http://localhost:8001';

export const api = {
  /**
   * List all conversations.
   */
  async listConversations() {
    const response = await fetch(`${API_BASE}/api/conversations`);
    if (!response.ok) {
      throw new Error('Failed to list conversations');
    }
    return response.json();
  },

  /**
   * Create a new conversation.
   */
  async createConversation(mode = 'baseline') {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ mode }),
    });
    if (!response.ok) {
      throw new Error('Failed to create conversation');
    }
    return response.json();
  },

  /**
   * Get a specific conversation.
   */
  async getConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}`
    );
    if (!response.ok) {
      throw new Error('Failed to get conversation');
    }
    return response.json();
  },

  /**
   * Send a message in a conversation.
   */
  async sendMessage(conversationId, content, manualContext = []) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content, manual_context: manualContext }),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to send message');
    }
    return response.json();
  },

  /**
   * Send a message and receive streaming updates.
   * @param {string} conversationId - The conversation ID
   * @param {string} content - The message content
   * @param {function} onEvent - Callback function for each event: (eventType, data) => void
   * @returns {Promise<void>}
   */
  async sendMessageStream(conversationId, content, manualContext = [], onEvent, signal) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content, manual_context: manualContext }),
        signal,
      }
    );

    if (!response.ok) {
      throw new Error('Failed to send message');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);
            onEvent(event.type, event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e);
          }
        }
      }
    }
  },

  /**
   * Upload and index a repository ZIP for a conversation.
   */
  async uploadRepo(conversationId, file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/upload_repo`,
      {
        method: 'POST',
        body: formData,
      }
    );

    const contentType = response.headers.get('content-type') || '';
    let data;

    // Prefer JSON payloads; fall back to text so we surface meaningful errors.
    if (contentType.includes('application/json')) {
      data = await response.json();
    } else {
      const text = await response.text();
      throw new Error(
        text || `Upload failed with status ${response.status}`
      );
    }

    if (!response.ok) {
      throw new Error(data?.message || 'Upload failed');
    }

    return data;
  },

  async getRepoTree(conversationId) {
    const response = await fetch(`${API_BASE}/api/conversations/${conversationId}/repo_tree`);
    if (!response.ok) {
      throw new Error('Failed to load repo tree');
    }
    return response.json();
  },

  async getFile(conversationId, path) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/file?path=${encodeURIComponent(path)}`
    );
    if (!response.ok) {
      throw new Error('Failed to load file');
    }
    return response.json();
  },

  async resolvePath(conversationId, query, userQuery = '') {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/resolve_path?q=${encodeURIComponent(query)}&user_query=${encodeURIComponent(userQuery)}`
    );
    if (!response.ok) {
      throw new Error('Failed to resolve path');
    }
    return response.json();
  },

  async searchRepo(conversationId, query, limit = 3) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/search?q=${encodeURIComponent(query)}&limit=${limit}`
    );
    if (!response.ok) {
      throw new Error('Failed to search repo');
    }
    return response.json();
  },

  async reindexGit(conversationId) {
    const response = await fetch(`${API_BASE}/api/conversations/${conversationId}/reindex_git`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error('Failed to reindex from git');
    }
    return response.json();
  },

  async getSettings() {
    const response = await fetch(`${API_BASE}/api/settings`);
    if (!response.ok) {
      throw new Error('Failed to load settings');
    }
    return response.json();
  },

  async updateSettings(payload) {
    const response = await fetch(`${API_BASE}/api/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error('Failed to update settings');
    }
    return response.json();
  },
};
