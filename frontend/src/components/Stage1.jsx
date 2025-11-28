import { useState, useEffect, forwardRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage1.css';

const Stage1 = forwardRef(function Stage1({ responses, focusedModel, onActiveChange }, ref) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  useEffect(() => {
    if (!focusedModel) return;
    const idx = responses.findIndex((r) => r.model === focusedModel);
    if (idx !== -1 && idx !== activeTab) {
      setActiveTab(idx);
    }
  }, [focusedModel, responses, activeTab]);

  const handleTab = (index) => {
    setActiveTab(index);
    onActiveChange && onActiveChange(responses[index].model);
  };

  return (
    <div className="stage stage1" ref={ref}>
      <h3 className="stage-title">Stage 1: Individual Responses</h3>

      <div className="tabs">
        {responses.map((resp, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => handleTab(index)}
          >
            {resp.model.split('/')[1] || resp.model}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="model-name">
          {responses[activeTab].model}
          {responses[activeTab].role && (
            <span className="role-tag">{responses[activeTab].role}</span>
          )}
        </div>
        <div className="response-text markdown-content">
          <ReactMarkdown>{responses[activeTab].response}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
});

export default Stage1;
