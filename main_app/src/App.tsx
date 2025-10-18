import { useState } from 'react';
import { ChatbotTab } from './components/ChatbotTab';
import { FinancesTab } from './components/FinancesTab';
import { ProfileTab } from './components/ProfileTab';
import { MessageSquare, PieChart, User } from 'lucide-react';
import { ThemeProvider } from './contexts/ThemeContext';

function AppContent() {
  const [activeTab, setActiveTab] = useState<'chatbot' | 'finances' | 'profile'>('chatbot');

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {activeTab === 'chatbot' && <ChatbotTab />}
        {activeTab === 'finances' && <FinancesTab />}
        {activeTab === 'profile' && <ProfileTab />}
      </div>

      {/* Bottom Navigation */}
      <nav className="border-t bg-white dark:bg-gray-800 dark:border-gray-700">
        <div className="flex justify-around items-center h-20">
          <button
            onClick={() => setActiveTab('chatbot')}
            className={`flex flex-col items-center justify-center flex-1 h-full transition-colors ${
              activeTab === 'chatbot' 
                ? 'text-[#2D9A86]' 
                : 'text-gray-400 dark:text-gray-500'
            }`}
          >
            <MessageSquare className="w-6 h-6 mb-1" />
            <span className="text-xs">Chatbot</span>
          </button>
          
          <button
            onClick={() => setActiveTab('finances')}
            className={`flex flex-col items-center justify-center flex-1 h-full transition-colors ${
              activeTab === 'finances' 
                ? 'text-[#2D9A86]' 
                : 'text-gray-400 dark:text-gray-500'
            }`}
          >
            <PieChart className="w-6 h-6 mb-1" />
            <span className="text-xs">Finances</span>
          </button>
          
          <button
            onClick={() => setActiveTab('profile')}
            className={`flex flex-col items-center justify-center flex-1 h-full transition-colors ${
              activeTab === 'profile' 
                ? 'text-[#2D9A86]' 
                : 'text-gray-400 dark:text-gray-500'
            }`}
          >
            <User className="w-6 h-6 mb-1" />
            <span className="text-xs">Profile</span>
          </button>
        </div>
      </nav>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}
