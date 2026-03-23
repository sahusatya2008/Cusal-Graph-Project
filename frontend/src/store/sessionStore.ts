import { create } from 'zustand';
import type { StateCreator } from 'zustand';

// Generate a unique session ID
const generateSessionId = () => {
  return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
};

interface SessionState {
  sessionId: string;
  datasetId: string | null;
  modelId: string | null;
  setSessionId: (id: string) => void;
  setDatasetId: (id: string | null) => void;
  setModelId: (id: string | null) => void;
}

type SessionStore = StateCreator<SessionState>;

const sessionStore: SessionStore = (set) => ({
  sessionId: generateSessionId(),
  datasetId: null,
  modelId: null,
  setSessionId: (id: string) => set({ sessionId: id }),
  setDatasetId: (id: string | null) => set({ datasetId: id }),
  setModelId: (id: string | null) => set({ modelId: id }),
});

export const useSessionStore = create<SessionState>(sessionStore);
