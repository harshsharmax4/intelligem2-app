
/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, Part, Content } from '@google/genai';
import * as marked from 'marked';

// --- Constants ---
const STORAGE_KEY = 'gemini_chat_history_v7';
const CONFIG_STORAGE_KEY = 'gemini_dev_config_v7';
const DEV_MODE_KEY = 'gemini_is_dev_mode_v7';

// --- Initialization ---
const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  console.error("API_KEY not found in environment.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

// --- Types ---
type Role = 'user' | 'model';
type Message = { role: Role; parts: Part[]; timestamp: number; mode?: string; metadata?: any };
type AppState = 'idle' | 'chatting' | 'thinking' | 'searching' | 'analyzing' | 'developer';

// --- State ---
let localHistory: Message[] = [];
let currentMedia: { data: string; mimeType: string } | null = null;
let appState: AppState = 'idle';
let lastResponseContent = ""; 
let isDevMode = false;

// Dev Mode State
let devConfig = {
  systemInstruction: "",
  temperature: 1.0,
  topP: 0.95,
  maxOutputTokens: 2048,
  model: 'gemini-2.5-flash-lite',
  safetyThreshold: 'BLOCK_MEDIUM_AND_ABOVE',
  tools: {
    googleSearch: false,
    googleMaps: false,
    thinking: false
  },
  forceTools: false 
};

// Load persistent data
try {
  const storedHistory = localStorage.getItem(STORAGE_KEY);
  if (storedHistory) {
    localHistory = JSON.parse(storedHistory);
    if (localHistory.length > 0) appState = 'chatting';
  }
} catch (e) {
  console.error("Failed to load history:", e);
}

try {
  const storedConfig = localStorage.getItem(CONFIG_STORAGE_KEY);
  if (storedConfig) {
    devConfig = { ...devConfig, ...JSON.parse(storedConfig) };
  }
} catch (e) {
  console.error("Failed to load dev config:", e);
}

try {
  const storedDevMode = localStorage.getItem(DEV_MODE_KEY);
  if (storedDevMode) {
    isDevMode = JSON.parse(storedDevMode);
  }
} catch (e) {
  console.error("Failed to load dev mode state:", e);
}

// --- Logic: Intent & Context Engine ---

const INTENTS = {
  SEARCH: /search|find|latest|news|google|weather|price|stock/i,
  THINK: /think|reason|plan|complex|solve|math|optimize|architect|code/i,
};

function determineConfig(text: string, hasMedia: boolean) {
  let model = 'gemini-2.5-flash-lite';
  let config: any = {};
  let modeLabel = 'Gemini Flash Lite';
  let visualState = 'glow-active';

  // 1. Tool Logic (Precedence: Dev Force > Auto-Detect)
  const useSearch = isDevMode && devConfig.forceTools ? devConfig.tools.googleSearch : INTENTS.SEARCH.test(text);
  const useThinking = isDevMode && devConfig.forceTools ? devConfig.tools.thinking : INTENTS.THINK.test(text);
  const useMaps = isDevMode && devConfig.forceTools ? devConfig.tools.googleMaps : false;

  if (hasMedia) {
    model = 'gemini-3-pro-preview';
    modeLabel = currentMedia?.mimeType.startsWith('video') ? 'Video Intelligence' : 'Visual Analysis';
    visualState = 'glow-vision';
  } else if (useThinking) {
    model = 'gemini-3-pro-preview';
    config.thinkingConfig = { thinkingBudget: 32768 };
    modeLabel = 'Deep Thinking';
    visualState = 'glow-think';
  } else if (useSearch || useMaps) {
    model = 'gemini-3-flash-preview';
    const tools: any[] = [];
    if (useSearch) tools.push({ googleSearch: {} });
    if (useMaps) tools.push({ googleMaps: {} });
    config.tools = tools;
    modeLabel = useSearch ? 'Google Search' : 'Google Maps';
    visualState = useSearch ? 'glow-search' : 'glow-maps';
  } else if (isDevMode) {
    model = devConfig.model;
    if (model === 'gemini-3-pro-preview') modeLabel = 'Gemini Pro';
    else if (model.includes('lite')) modeLabel = 'Gemini Lite';
    else modeLabel = 'Gemini Custom';
  }

  // 2. Dev Mode Configuration Overrides
  if (isDevMode) {
    if (devConfig.systemInstruction.trim()) {
      config.systemInstruction = devConfig.systemInstruction;
    }
    
    config.temperature = devConfig.temperature;
    config.topP = devConfig.topP;
    
    // Feature Constraint: Do not set maxOutputTokens when using thinking mode
    if (!config.thinkingConfig) {
      config.maxOutputTokens = devConfig.maxOutputTokens;
    }

    // Safety Settings Mapping
    const categories = [
      'HARM_CATEGORY_HARASSMENT',
      'HARM_CATEGORY_HATE_SPEECH',
      'HARM_CATEGORY_SEXUALLY_EXPLICIT',
      'HARM_CATEGORY_DANGEROUS_CONTENT'
    ];
    config.safetySettings = categories.map(cat => ({
      category: cat,
      threshold: devConfig.safetyThreshold
    }));

    modeLabel += ' (Dev)';
  }

  return { model, config, modeLabel, visualState };
}

// Simple "Anticipation Engine" to suggest actions
function getContextualActions(text: string, hasMedia: boolean, lastMsgRole?: Role): { label: string, prompt: string, icon: string }[] {
  const actions = [];

  if (hasMedia) {
    actions.push({ label: 'Describe', prompt: 'Describe this in detail.', icon: 'visibility' });
    if (currentMedia?.mimeType.startsWith('image')) {
      actions.push({ label: 'Extract Text', prompt: 'Extract all text from this image.', icon: 'text_fields' });
    }
    actions.push({ label: 'Analyze', prompt: 'Analyze the key elements.', icon: 'analytics' });
    return actions; 
  }

  if (text.length > 0) {
    if (text.includes('code') || text.includes('function') || text.includes('const ')) {
      actions.push({ label: 'Fix Bugs', prompt: 'Find and fix bugs in this code.', icon: 'bug_report' });
      actions.push({ label: 'Explain', prompt: 'Explain this code step by step.', icon: 'description' });
    }
    else if (INTENTS.SEARCH.test(text)) {
      actions.push({ label: 'Deep Dive', prompt: 'Give me a detailed deep dive on this.', icon: 'scuba_diving' });
      actions.push({ label: 'Fact Check', prompt: 'Verify this information.', icon: 'fact_check' });
    }
    else if (INTENTS.THINK.test(text)) {
        actions.push({ label: 'Break Down', prompt: 'Break this problem down step-by-step.', icon: 'segment' });
    }

    if (actions.length === 0) {
        actions.push({ label: 'Improve', prompt: 'Improve this writing.', icon: 'edit_note' });
        actions.push({ label: 'Creative Twist', prompt: 'Add a creative twist.', icon: 'auto_awesome' });
        actions.push({ label: 'Expand', prompt: 'Expand on this idea.', icon: 'open_in_full' });
    }

    return actions;
  }

  if (lastMsgRole === 'model' && lastResponseContent) {
    if (lastResponseContent.includes('```')) { 
      actions.push({ label: 'Refactor', prompt: 'Refactor this code for better performance.', icon: 'build' });
      actions.push({ label: 'Add Comments', prompt: 'Add detailed comments to the code.', icon: 'comment' });
    } else if (lastResponseContent.length > 500) {
      actions.push({ label: 'Summarize', prompt: 'Summarize the above in 3 bullet points.', icon: 'short_text' });
    }
    actions.push({ label: 'Verify', prompt: 'Are you sure? Verify this information.', icon: 'check_circle' });
  }

  if (actions.length === 0) {
    actions.push({ label: 'Brainstorm', prompt: 'Help me brainstorm creative ideas.', icon: 'lightbulb' });
    actions.push({ label: 'Plan', prompt: 'Help me create a structured plan.', icon: 'calendar_today' });
  }

  return actions.slice(0, 4);
}

// --- DOM & Component ---

const root = document.getElementById('root');

function App() {
  const hour = new Date().getHours();
  let themeClass = 'theme-night';
  if (hour >= 6 && hour < 17) themeClass = 'theme-day';
  else if (hour >= 17 && hour < 21) themeClass = 'theme-dusk';

  const container = document.createElement('div');
  container.className = `chat-container state-${appState} ${themeClass}`;

  // Gooey SVG Filter definition
  const gooeyFilter = document.createElement('div');
  gooeyFilter.style.height = '0';
  gooeyFilter.style.width = '0';
  gooeyFilter.style.position = 'absolute';
  gooeyFilter.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" version="1.1">
      <defs>
        <filter id="goo">
          <feGaussianBlur in="SourceGraphic" stdDeviation="15" result="blur" />
          <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7" result="goo" />
          <feBlend in="SourceGraphic" in2="goo" />
        </filter>
      </defs>
    </svg>
  `;
  container.appendChild(gooeyFilter);

  // 1. Welcome View
  const welcomeView = document.createElement('div');
  welcomeView.className = 'welcome-view';
  welcomeView.innerHTML = `
    <div class="liquid-bg-container">
      <div class="sun-rays"></div>
      <div class="liquid-orb orb-1"></div>
      <div class="liquid-orb orb-2"></div>
      <div class="liquid-orb orb-3"></div>
    </div>
    <div class="glass-panes-container">
       <div class="glass-blob blob-1"></div>
       <div class="glass-blob blob-2"></div>
       <div class="glass-blob blob-3"></div>
       <div class="glass-blob blob-4"></div>
    </div>
    <div class="content-grid-layer">
      <div class="grid-item item-1" data-prompt="Find the latest developments in AI" data-tooltip="Search current news and general knowledge">
          <div class="card-glass">
            <div class="card-icon">search</div>
            <div class="card-main-text">Search Web</div>
            <div class="card-sub-text">Get up-to-date info</div>
          </div>
      </div>
      <div class="grid-item item-2" data-prompt="Analyze this image for design patterns" data-tooltip="Describe and extract insights from images or videos">
          <div class="card-glass">
            <div class="card-icon">image</div>
            <div class="card-main-text">Visual Analysis</div>
            <div class="card-sub-text">Understand images</div>
          </div>
      </div>
      <div class="grid-item item-3" data-prompt="Architect a scalable backend system" data-tooltip="Chain-of-thought reasoning for complex problems">
          <div class="card-glass">
            <div class="card-icon">psychology</div>
            <div class="card-main-text">Deep Thinking</div>
            <div class="card-sub-text">Complex reasoning</div>
          </div>
      </div>
      <div class="grid-item item-4" data-action="toggle-dev" data-tooltip="Adjust model parameters and persona">
          <div class="card-glass">
            <div class="card-icon dev-icon">terminal</div>
            <div class="card-main-text">Developer Mode</div>
            <div class="card-sub-text">Configure Persona & Params</div>
          </div>
      </div>
    </div>
    <div class="welcome-header">
      <div class="welcome-icon">
         <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
           <line x1="12" y1="5" x2="12" y2="19"></line>
           <line x1="5" y1="12" x2="19" y2="12"></line>
         </svg>
      </div>
      <h1>What can I do for you?</h1>
      <p>I can search, analyze images, or help you think through complex problems.</p>
    </div>
  `;

  // 2. Chat History
  const historyDiv = document.createElement('div');
  historyDiv.className = 'chat-history';
  historyDiv.id = 'chat-history';

  // 3. Dev Console
  const devConsole = document.createElement('div');
  devConsole.className = 'dev-console';
  if (isDevMode) devConsole.classList.add('visible');
  devConsole.innerHTML = `
    <div class="dev-header">
      <span><span class="material-icons">terminal</span> Developer Console</span>
      <button class="close-dev" data-tooltip="Close console">×</button>
    </div>
    <div class="dev-body">
      <div class="dev-tabs">
        <button class="dev-tab-btn active" data-target="dev-tab-gen" data-tooltip="General configuration">General</button>
        <button class="dev-tab-btn" data-target="dev-tab-tools" data-tooltip="Grounding & logic tools">Tools</button>
        <button class="dev-tab-btn" data-target="dev-tab-safety" data-tooltip="Content filtering levels">Safety</button>
      </div>

      <div class="dev-tab-content active" id="dev-tab-gen">
        <div class="dev-row">
          <label data-tooltip="Permanent instructions for the model's behavior">System Instruction (Persona)</label>
          <textarea id="sys-instruction" placeholder="Enter system instructions...">${devConfig.systemInstruction}</textarea>
        </div>
        <div class="dev-grid">
          <div class="dev-row">
            <label data-tooltip="Base model for standard requests">Base Model</label>
            <select id="model-select">
              <option value="gemini-2.5-flash-lite" ${devConfig.model === 'gemini-2.5-flash-lite' ? 'selected' : ''}>Flash Lite</option>
              <option value="gemini-3-flash-preview" ${devConfig.model === 'gemini-3-flash-preview' ? 'selected' : ''}>Flash 3</option>
              <option value="gemini-3-pro-preview" ${devConfig.model === 'gemini-3-pro-preview' ? 'selected' : ''}>Pro 3</option>
            </select>
          </div>
          <div class="dev-row">
            <label data-tooltip="Limit response length (ignored in Deep Thinking mode)">Max Tokens (<span id="token-val">${devConfig.maxOutputTokens}</span>)</label>
            <input type="number" id="token-input" value="${devConfig.maxOutputTokens}" step="128" min="1">
          </div>
          <div class="dev-row">
            <label data-tooltip="Lower for factual answers, higher for creative ones">Temperature (<span id="temp-val">${devConfig.temperature.toFixed(1)}</span>)</label>
            <input type="range" id="temp-range" min="0" max="2" step="0.1" value="${devConfig.temperature}">
          </div>
          <div class="dev-row">
            <label data-tooltip="Controls vocabulary diversity">Top P (<span id="topp-val">${devConfig.topP.toFixed(2)}</span>)</label>
            <input type="range" id="topp-range" min="0" max="1" step="0.05" value="${devConfig.topP}">
          </div>
        </div>
      </div>

      <div class="dev-tab-content" id="dev-tab-tools">
        <div class="dev-row">
          <label style="display:flex; justify-content:space-between; align-items:center;" data-tooltip="Ignore automatic intent detection and use manually selected tools">
             Force Programmatic Tools
             <input type="checkbox" id="force-tools-toggle" style="width:20px; height:20px;" ${devConfig.forceTools ? 'checked' : ''}>
          </label>
          <p style="font-size:11px; color:#666; margin-top:-5px;">Override auto-intent detection with manual selection.</p>
        </div>
        <div class="tool-toggles">
          <div class="tool-item" data-tooltip="Ground responses in live Google Search results">
            <span class="material-icons">search</span>
            <span>Google Search</span>
            <input type="checkbox" id="tool-search" class="ios-toggle" ${devConfig.tools.googleSearch ? 'checked' : ''}>
          </div>
          <div class="tool-item" data-tooltip="Access location and place information">
            <span class="material-icons">map</span>
            <span>Google Maps</span>
            <input type="checkbox" id="tool-maps" class="ios-toggle" ${devConfig.tools.googleMaps ? 'checked' : ''}>
          </div>
          <div class="tool-item" data-tooltip="Enables 32k token thinking budget (Gemini 3 Pro only)">
            <span class="material-icons">psychology</span>
            <span>Deep Thinking</span>
            <input type="checkbox" id="tool-think" class="ios-toggle" ${devConfig.tools.thinking ? 'checked' : ''}>
          </div>
        </div>
      </div>

      <div class="dev-tab-content" id="dev-tab-safety">
        <div class="dev-row">
          <label data-tooltip="Sensitivity for Harassment, Hate, Sexual, and Dangerous content">Harm Threshold</label>
          <select id="safety-select">
            <option value="BLOCK_NONE" ${devConfig.safetyThreshold === 'BLOCK_NONE' ? 'selected' : ''}>Allow All (Block None)</option>
            <option value="BLOCK_ONLY_HIGH" ${devConfig.safetyThreshold === 'BLOCK_ONLY_HIGH' ? 'selected' : ''}>Block High Only</option>
            <option value="BLOCK_MEDIUM_AND_ABOVE" ${devConfig.safetyThreshold === 'BLOCK_MEDIUM_AND_ABOVE' ? 'selected' : ''}>Block Medium & Above</option>
            <option value="BLOCK_LOW_AND_ABOVE" ${devConfig.safetyThreshold === 'BLOCK_LOW_AND_ABOVE' ? 'selected' : ''}>Block Low & Above</option>
          </select>
          <p style="font-size:11px; color:#666; margin-top:8px;">Applies to all harm categories.</p>
        </div>
      </div>
    </div>
  `;

  // 4. Input Area
  const inputArea = document.createElement('div');
  inputArea.className = 'input-area';
  const quickActionsBar = document.createElement('div');
  quickActionsBar.className = 'quick-actions-bar';
  inputArea.appendChild(quickActionsBar);

  const inputWrapper = document.createElement('div');
  inputWrapper.className = 'input-wrapper';
  const inputContent = document.createElement('div');
  inputContent.className = 'input-content';
  const modePill = document.createElement('div');
  modePill.className = 'mode-pill';
  inputContent.appendChild(modePill);

  const dragOverlay = document.createElement('div');
  dragOverlay.className = 'drag-overlay';
  dragOverlay.innerHTML = '<span style="font-size: 24px; margin-bottom: 8px;">📂</span><span>Drop to Analyze</span>';
  inputWrapper.appendChild(dragOverlay);

  const mediaPreviewContainer = document.createElement('div');
  mediaPreviewContainer.className = 'media-preview-container';
  inputContent.appendChild(mediaPreviewContainer);

  const inputRow = document.createElement('div');
  inputRow.className = 'input-row';
  const attachBtn = document.createElement('button');
  attachBtn.className = 'action-btn';
  attachBtn.setAttribute('data-tooltip', 'Upload images or video (drag & drop supported)');
  attachBtn.innerHTML = `<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>`;
  attachBtn.onclick = () => fileInput.click();

  const devToggleBtn = document.createElement('button');
  devToggleBtn.className = `dev-toggle-badge ${isDevMode ? 'active' : ''}`;
  devToggleBtn.setAttribute('data-tooltip', isDevMode ? 'Close Developer Console' : 'Open Developer Console');
  devToggleBtn.innerHTML = `<span class="material-icons" style="font-size: 16px;">terminal</span> terminal`;

  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/*,video/*';
  fileInput.style.display = 'none';
  
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = isDevMode ? "Enter prompt (Dev Mode Active)..." : "Type a message...";
  input.id = 'prompt-input';
  input.autocomplete = 'off';

  const sendBtn = document.createElement('button');
  sendBtn.className = 'send-btn';
  sendBtn.setAttribute('data-tooltip', 'Send message');
  sendBtn.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>`;

  inputRow.appendChild(attachBtn);
  inputRow.appendChild(devToggleBtn);
  inputRow.appendChild(input);
  inputRow.appendChild(sendBtn);
  inputContent.appendChild(inputRow);
  
  inputWrapper.appendChild(inputContent);
  inputArea.appendChild(inputWrapper);

  container.appendChild(welcomeView);
  container.appendChild(historyDiv);
  container.appendChild(inputArea);
  // Re-ordered to ensure DevConsole sits on top if z-index is fighting context
  container.appendChild(devConsole);
  container.appendChild(fileInput);

  // --- Functions ---

  const toggleDevMode = (forceState?: boolean) => {
    isDevMode = forceState !== undefined ? forceState : !isDevMode;
    // Persist Dev Mode State
    try { localStorage.setItem(DEV_MODE_KEY, JSON.stringify(isDevMode)); } catch (e) {}

    if (isDevMode) {
      devConsole.classList.add('visible');
      devToggleBtn.classList.add('active');
      devToggleBtn.setAttribute('data-tooltip', 'Close Developer Console');
      input.placeholder = "Enter prompt (Dev Mode Active)...";
    } else {
      devConsole.classList.remove('visible');
      devToggleBtn.classList.remove('active');
      devToggleBtn.setAttribute('data-tooltip', 'Open Developer Console');
      input.placeholder = "Type a message...";
    }
    updateUIForInput();
  };

  const updateDevConfig = () => {
    // CRITICAL FIX: Use scoped querySelector on devConsole instead of document.getElementById
    // to ensure elements are found even if not yet fully in DOM or if in shadow DOM scenarios.
    const sys = (devConsole.querySelector('#sys-instruction') as HTMLTextAreaElement).value;
    const model = (devConsole.querySelector('#model-select') as HTMLSelectElement).value;
    const temp = parseFloat((devConsole.querySelector('#temp-range') as HTMLInputElement).value);
    const topp = parseFloat((devConsole.querySelector('#topp-range') as HTMLInputElement).value);
    const tokens = parseInt((devConsole.querySelector('#token-input') as HTMLInputElement).value);
    const forceTools = (devConsole.querySelector('#force-tools-toggle') as HTMLInputElement).checked;
    const safety = (devConsole.querySelector('#safety-select') as HTMLSelectElement).value;

    const tSearch = (devConsole.querySelector('#tool-search') as HTMLInputElement).checked;
    const tMaps = (devConsole.querySelector('#tool-maps') as HTMLInputElement).checked;
    const tThink = (devConsole.querySelector('#tool-think') as HTMLInputElement).checked;
    
    devConfig = { 
      ...devConfig, 
      systemInstruction: sys, 
      model, 
      temperature: temp, 
      topP: topp, 
      maxOutputTokens: tokens,
      forceTools: forceTools,
      safetyThreshold: safety,
      tools: {
        googleSearch: tSearch,
        googleMaps: tMaps,
        thinking: tThink
      }
    };

    (devConsole.querySelector('#temp-val') as HTMLElement).textContent = temp.toFixed(1);
    (devConsole.querySelector('#topp-val') as HTMLElement).textContent = topp.toFixed(2);
    (devConsole.querySelector('#token-val') as HTMLElement).textContent = tokens.toString();
    
    // Persist config
    try { localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(devConfig)); } catch (e) {}

    updateUIForInput();
  };

  const updateQuickActions = () => {
    const text = input.value;
    const hasMedia = !!currentMedia;
    const lastMsg = localHistory.length > 0 ? localHistory[localHistory.length - 1] : undefined;
    const actions = getContextualActions(text, hasMedia, lastMsg?.role);
    
    quickActionsBar.innerHTML = '';
    if (actions.length > 0) {
      quickActionsBar.classList.add('visible');
      actions.forEach(action => {
        const chip = document.createElement('button');
        chip.className = 'action-chip';
        chip.setAttribute('data-tooltip', `Run: "${action.prompt}"`);
        chip.innerHTML = `<span class="material-icons">${action.icon}</span> ${action.label}`;
        chip.onclick = () => {
           input.value = action.prompt;
           handleSend();
        };
        quickActionsBar.appendChild(chip);
      });
    } else {
      quickActionsBar.classList.remove('visible');
    }
  };

  const updateUIForInput = () => {
    const text = input.value;
    const hasMedia = !!currentMedia;
    const { modeLabel, visualState } = determineConfig(text, hasMedia);

    inputWrapper.className = `input-wrapper ${text.length > 0 || hasMedia || isDevMode ? visualState : ''}`;
    
    if (text.length > 2 || hasMedia || isDevMode) {
      modePill.innerHTML = `<span>Active Model</span> ${modeLabel}`;
      modePill.classList.add('visible');
      sendBtn.classList.add('active');
    } else {
      modePill.classList.remove('visible');
      sendBtn.classList.remove('active');
    }

    updateQuickActions();
  };

  const clearMedia = () => {
    currentMedia = null;
    mediaPreviewContainer.classList.remove('visible');
    mediaPreviewContainer.innerHTML = '';
    updateUIForInput();
  };

  const handleFile = (file: File) => {
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');
    if (!isImage && !isVideo) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      currentMedia = { data: result.split(',')[1], mimeType: file.type };
      mediaPreviewContainer.classList.add('visible');
      mediaPreviewContainer.innerHTML = `
        <div class="preview-wrapper">
          ${isImage 
            ? `<img src="${result}" class="preview-thumb"/>` 
            : `<div class="msg-video-badge" style="height: 100%; display: flex; align-items: center; justify-content: center; font-size: 24px;">📹</div>`
          }
          <div class="remove-media" data-tooltip="Remove file">×</div>
        </div>
      `;
      mediaPreviewContainer.querySelector('.remove-media')?.addEventListener('click', (e) => {
        e.stopPropagation();
        clearMedia();
      });
      updateUIForInput();
    };
    reader.readAsDataURL(file);
  };

  const renderAgentStatus = (label: string, container: HTMLElement) => {
    const statusDiv = document.createElement('div');
    statusDiv.className = 'agent-status';
    let color = '#888';
    if (label.includes('Think')) color = '#AF52DE';
    else if (label.includes('Search')) color = '#34C759';
    else if (label.includes('Maps')) color = '#FF9500';
    else if (label.includes('Video') || label.includes('Vis')) color = '#FF2D55';
    else if (label.includes('Dev')) color = '#FF3B30';

    statusDiv.innerHTML = `
      <div class="agent-status-icon">
        <div class="status-dot" style="color: ${color}"></div>
      </div>
      <span>${label}...</span>
    `;
    container.appendChild(statusDiv);
    return statusDiv;
  };

  const handleSend = async () => {
    const text = input.value.trim();
    if (!text && !currentMedia) return;

    appState = 'chatting';
    container.className = `chat-container state-chatting ${themeClass}`;
    welcomeView.style.display = 'none';
    historyDiv.style.display = 'flex';
    historyDiv.classList.add('active');

    const { model, config, modeLabel } = determineConfig(text, !!currentMedia);
    
    if (modeLabel.includes('Think')) container.classList.add('state-thinking');
    else if (modeLabel.includes('Search')) container.classList.add('state-searching');
    else if (modeLabel.includes('Video') || modeLabel.includes('Vis')) container.classList.add('state-analyzing');
    else if (isDevMode) container.classList.add('state-dev');

    const parts: Part[] = [];
    if (currentMedia) {
      parts.push({ inlineData: { data: currentMedia.data, mimeType: currentMedia.mimeType } });
    }
    if (text) parts.push({ text: text });

    input.value = '';
    clearMedia();
    updateUIForInput();

    const userMsg: Message = { role: 'user', parts: parts, timestamp: Date.now() };
    addMessageToDOM(userMsg);
    saveMessage(userMsg);

    const aiMessageDiv = createMessageDiv('model', modeLabel);
    const contentDiv = aiMessageDiv.querySelector('.message-content') as HTMLElement;
    historyDiv.appendChild(aiMessageDiv);
    
    const statusEl = renderAgentStatus(
       modeLabel === 'Gemini Flash Lite' ? 'Generating' : modeLabel, 
       contentDiv
    );
    scrollToBottom();

    try {
      const historyForApi: Content[] = localHistory.map(m => ({
        role: m.role,
        parts: m.parts
      }));

      const chat = ai.chats.create({
        model: model,
        history: historyForApi,
        config: config
      });

      const result = await chat.sendMessageStream({ 
        message: parts.length === 1 && parts[0].text ? parts[0].text : parts 
      });
      
      let fullResponseText = '';
      let isFirstChunk = true;

      for await (const chunk of result) {
        if (isFirstChunk) {
           statusEl.remove();
           isFirstChunk = false;
        }

        fullResponseText += chunk.text;
        const chunkAny = chunk as any;
        const parsedHTML = await Promise.resolve(marked.parse(fullResponseText));
        contentDiv.innerHTML = parsedHTML;
        
        if (chunkAny.candidates?.[0]?.groundingMetadata) {
          renderGroundingSources(chunkAny.candidates[0].groundingMetadata, contentDiv);
        }
        enhanceMessageContent(contentDiv);
        scrollToBottom();
      }

      lastResponseContent = fullResponseText;
      saveMessage({
        role: 'model',
        parts: [{ text: fullResponseText }],
        timestamp: Date.now(),
        mode: modeLabel
      });
      updateQuickActions();
    } catch (e) {
      console.error(e);
      statusEl.innerHTML = `<span style="color:#ff453a">Error: ${(e as Error).message}</span>`;
      statusEl.style.animation = 'none';
      statusEl.style.borderColor = '#ff453a';
    } finally {
      container.className = `chat-container state-chatting ${themeClass}`;
    }
  };

  function renderGroundingSources(metadata: any, container: HTMLElement) {
    let sourcesContainer = container.querySelector('.sources-container');
    if (!sourcesContainer) {
      sourcesContainer = document.createElement('div');
      sourcesContainer.className = 'sources-container';
      sourcesContainer.innerHTML = '<div class="sources-label" style="font-size:12px;color:#888;margin-bottom:8px;">SOURCES</div><div class="sources-grid"></div>';
      container.appendChild(sourcesContainer);
    }
    const grid = sourcesContainer.querySelector('.sources-grid');
    if (!grid) return;

    const chunks = metadata.groundingChunks || [];
    const webSources = chunks.filter((c: any) => c.web).map((c: any) => c.web);
    const mapSources = chunks.filter((c: any) => c.maps).map((c: any) => c.maps);

    grid.innerHTML = ''; 
    [...webSources, ...mapSources].forEach((s: any) => {
      const card = document.createElement('a');
      card.className = 'source-card';
      card.href = s.uri;
      card.target = '_blank';
      card.setAttribute('data-tooltip', `Open source: ${s.title || 'External link'}`);
      card.innerHTML = `
        <span class="material-icons" style="font-size:14px;color:#888;">${s.maps ? 'place' : 'public'}</span>
        <div class="source-title">${s.title || s.uri}</div>
      `;
      grid.appendChild(card);
    });
  }

  // --- Event Listeners ---
  sendBtn.addEventListener('click', handleSend);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleSend();
  });
  input.addEventListener('input', updateUIForInput);
  fileInput.addEventListener('change', (e) => {
    if (fileInput.files?.length) handleFile(fileInput.files[0]);
  });

  container.addEventListener('click', (e) => {
    const item = (e.target as HTMLElement).closest('.grid-item');
    if (item) {
      const prompt = item.getAttribute('data-prompt');
      const action = item.getAttribute('data-action');
      if (action === 'toggle-dev') { toggleDevMode(true); return; }
      if (prompt) {
        input.value = prompt;
        updateUIForInput();
        handleSend();
      }
    }
  });

  devToggleBtn.addEventListener('click', () => toggleDevMode());
  devConsole.querySelector('.close-dev')?.addEventListener('click', () => {
    toggleDevMode(false);
  });

  // Tab switching
  devConsole.querySelectorAll('.dev-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      devConsole.querySelectorAll('.dev-tab-btn').forEach(b => b.classList.remove('active'));
      devConsole.querySelectorAll('.dev-tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.getAttribute('data-target');
      devConsole.querySelector(`#${target}`)?.classList.add('active');
    });
  });
  
  // Listeners for config changes - CRITICAL FIX: Attach to elements directly in scope
  const configIds = ['sys-instruction', 'model-select', 'temp-range', 'topp-range', 'token-input', 'force-tools-toggle', 'safety-select', 'tool-search', 'tool-maps', 'tool-think'];
  configIds.forEach(id => {
    const el = devConsole.querySelector(`#${id}`);
    el?.addEventListener('input', updateDevConfig);
    el?.addEventListener('change', updateDevConfig);
  });

  document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (items) {
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1 || items[i].type.indexOf('video') !== -1) {
          const blob = items[i].getAsFile();
          if (blob) handleFile(blob);
          e.preventDefault(); 
          break;
        }
      }
    }
  });

  // Robust Drag and Drop Logic
  let dragCounter = 0;
  const handleDrag = (e: DragEvent) => { e.preventDefault(); e.stopPropagation(); };
  
  container.addEventListener('dragenter', (e) => {
    handleDrag(e);
    dragCounter++;
    if (dragCounter === 1) {
      inputWrapper.classList.add('drag-active');
    }
  });
  
  container.addEventListener('dragleave', (e) => {
    handleDrag(e);
    dragCounter--;
    if (dragCounter <= 0) {
      dragCounter = 0;
      inputWrapper.classList.remove('drag-active');
    }
  });
  
  container.addEventListener('drop', (e) => {
    handleDrag(e);
    dragCounter = 0;
    inputWrapper.classList.remove('drag-active');
    if (e.dataTransfer?.files.length) handleFile(e.dataTransfer.files[0]);
  });

  if (localHistory.length > 0) {
    welcomeView.style.display = 'none';
    historyDiv.style.display = 'flex';
    historyDiv.classList.add('active');
    localHistory.forEach(msg => addMessageToDOM(msg));
    lastResponseContent = localHistory[localHistory.length - 1].role === 'model' 
      ? localHistory[localHistory.length - 1].parts[0].text || "" 
      : "";
    updateQuickActions();
  } else {
    updateQuickActions();
  }
  
  // Initialize UI if Dev Mode is Active on Load
  if (isDevMode) updateUIForInput();

  return container;
}

// --- Helpers ---

function saveMessage(msg: Message) {
  localHistory.push(msg);
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(localHistory)); } catch (e) {}
}

function createMessageDiv(role: Role, mode?: string) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message message-${role}`;
  const content = document.createElement('div');
  content.className = 'message-content';
  msgDiv.appendChild(content);
  return msgDiv;
}

function addMessageToDOM(msg: Message) {
  const history = document.getElementById('chat-history');
  const msgDiv = createMessageDiv(msg.role, msg.mode);
  const contentDiv = msgDiv.querySelector('.message-content') as HTMLElement;

  if (msg.role === 'user') {
    let html = '';
    msg.parts.forEach(p => {
      if (p.inlineData) {
        if (p.inlineData.mimeType.startsWith('image')) {
          html += `<img src="data:${p.inlineData.mimeType};base64,${p.inlineData.data}" class="msg-image" />`;
        } else if (p.inlineData.mimeType.startsWith('video')) {
          html += `<div class="msg-video-badge">📹 Video Attached</div>`;
        }
      }
      if (p.text) html += `<p>${p.text}</p>`;
    });
    contentDiv.innerHTML = html;
  } else {
    const renderer = new marked.Renderer();
    renderer.blockquote = ({ text }) => `<blockquote class="insight-card">${text}</blockquote>`;
    marked.use({ renderer });
    const parseResult = marked.parse(msg.parts[0].text || '');
    Promise.resolve(parseResult).then(html => {
      contentDiv.innerHTML = html;
      enhanceMessageContent(contentDiv);
    });
  }
  history?.appendChild(msgDiv);
}

function enhanceMessageContent(container: HTMLElement) {
  container.querySelectorAll('pre').forEach(pre => {
    if (!pre.parentElement?.classList.contains('mac-window')) {
        const wrapper = document.createElement('div');
        wrapper.className = 'mac-window';
        const header = document.createElement('div');
        header.className = 'mac-window-header';
        const btns = document.createElement('div');
        btns.className = 'mac-buttons';
        btns.innerHTML = `<div class="mac-btn red"></div><div class="mac-btn yellow"></div><div class="mac-btn green"></div>`;
        const copyBtn = document.createElement('button');
        copyBtn.className = 'code-copy-btn';
        copyBtn.textContent = 'Copy';
        copyBtn.setAttribute('data-tooltip', 'Copy code snippet to clipboard');
        copyBtn.onclick = () => {
            const code = pre.querySelector('code')?.textContent;
            if (code) {
                navigator.clipboard.writeText(code);
                copyBtn.textContent = 'Copied';
                setTimeout(() => copyBtn.textContent = 'Copy', 2000);
            }
        };
        header.appendChild(btns);
        header.appendChild(copyBtn);
        pre.parentNode?.insertBefore(wrapper, pre);
        wrapper.appendChild(header);
        wrapper.appendChild(pre);
    }
  });
}

function scrollToBottom() {
  const history = document.getElementById('chat-history');
  if (history) history.scrollTop = history.scrollHeight;
}

if (root) { root.appendChild(App()); }
