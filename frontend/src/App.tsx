import { useState } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { 
  Upload, 
  GitBranch, 
  Sliders, 
  GitCompare, 
  Target, 
  BookOpen, 
  FileText,
  Menu,
  X
} from 'lucide-react';
import DataUpload from './components/DataUpload';
import CausalGraphView from './components/CausalGraphView';
import InterventionPanel from './components/InterventionPanel';
import CounterfactualPanel from './components/CounterfactualPanel';
import OptimizationDashboard from './components/OptimizationDashboard';
import MathExplanation from './components/MathExplanation';
import ReportsView from './components/ReportsView';
import { useSessionStore } from './store/sessionStore';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { sessionId, setSessionId } = useSessionStore();

  // Initialize session on first load
  useState(() => {
    if (!sessionId) {
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      setSessionId(newSessionId);
    }
  });

  const navItems = [
    { to: '/', icon: Upload, label: 'Data Upload' },
    { to: '/graph', icon: GitBranch, label: 'Causal Graph' },
    { to: '/intervention', icon: Sliders, label: 'Interventions' },
    { to: '/counterfactual', icon: GitCompare, label: 'Counterfactuals' },
    { to: '/optimization', icon: Target, label: 'Optimization' },
    { to: '/math', icon: BookOpen, label: 'Math Guide' },
    { to: '/reports', icon: FileText, label: 'Reports' },
  ];

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 flex">
        {/* Sidebar */}
        <aside 
          className={`${
            sidebarOpen ? 'w-64' : 'w-16'
          } bg-gradient-to-b from-indigo-900 to-purple-900 text-white transition-all duration-300 flex flex-col`}
        >
          <div className="p-4 flex items-center justify-between border-b border-indigo-700">
            {sidebarOpen && (
              <h1 className="text-lg font-bold truncate">Causal Lab</h1>
            )}
            <button 
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-1 hover:bg-indigo-700 rounded"
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
          
          <nav className="flex-1 p-2">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-3 py-2.5 rounded-lg mb-1 transition-colors ${
                    isActive
                      ? 'bg-white/20 text-white'
                      : 'text-indigo-200 hover:bg-white/10 hover:text-white'
                  }`
                }
              >
                <item.icon size={20} />
                {sidebarOpen && <span>{item.label}</span>}
              </NavLink>
            ))}
          </nav>

          {sidebarOpen && (
            <div className="p-4 border-t border-indigo-700 text-xs text-indigo-300">
              <p>Session: {sessionId?.slice(0, 12)}...</p>
            </div>
          )}
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          <div className="p-6 max-w-7xl mx-auto">
            <Routes>
              <Route path="/" element={<DataUpload />} />
              <Route path="/graph" element={<CausalGraphView />} />
              <Route path="/intervention" element={<InterventionPanel />} />
              <Route path="/counterfactual" element={<CounterfactualPanel />} />
              <Route path="/optimization" element={<OptimizationDashboard />} />
              <Route path="/math" element={<MathExplanation />} />
              <Route path="/reports" element={<ReportsView />} />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;