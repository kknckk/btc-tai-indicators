'use client';
import React, { useState } from 'react';
import { MessageSquare, Send, Bot, Loader2, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export function ChatAgent() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<{role: 'user' | 'agent', text: string}[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/v1/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg })
      });
      const data = await res.json();
      
      setMessages(prev => [...prev, { 
        role: 'agent', 
        text: data.response || "Błąd parsowania odpowiedzi od asystenta."
      }]);
    } catch (err) {
      setMessages(prev => [...prev, { 
        role: 'agent', 
        text: "Wystąpił błąd po stronie klienta przy próbie kontaktu z API."
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <button 
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 bg-blue-600 hover:bg-blue-500 text-white p-4 rounded-full shadow-2xl transition-all flex items-center justify-center z-40"
      >
        <MessageSquare className="w-6 h-6" />
      </button>

      {isOpen && (
        <div className="fixed bottom-24 right-6 w-96 h-[500px] bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl flex flex-col z-50 overflow-hidden">
          <div className="bg-slate-800 p-4 border-b border-slate-700 flex justify-between items-center">
            <div className="flex items-center gap-2 text-white font-bold">
              <Bot className="text-blue-400" />
              Quant Analyst AI
            </div>
            <button onClick={() => setIsOpen(false)} className="text-slate-400 hover:text-white transition-colors">
              Zamknij
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
              <div className="text-slate-400 text-sm text-center mt-10">
                Zadaj pytanie dotyczące aktualnej sytuacji rynkowej (EWI, MVRV, kointegracje)
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'agent' && <div className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-blue-400 flex-shrink-0"><Bot size={18}/></div>}
                <div className={`p-3 rounded-xl max-w-[80%] text-sm overflow-x-auto ${
                  msg.role === 'user' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-slate-800 text-slate-200 rounded-bl-none'
                }`}>
                  {msg.role === 'agent' ? (
                     <div className="prose prose-invert prose-sm max-w-none"><ReactMarkdown>{msg.text}</ReactMarkdown></div>
                  ) : (
                    msg.text
                  )}
                </div>
                {msg.role === 'user' && <div className="w-8 h-8 rounded bg-blue-700 flex items-center justify-center text-white flex-shrink-0"><User size={18}/></div>}
              </div>
            ))}
            {isLoading && (
               <div className="flex gap-3 justify-start">
                  <div className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-blue-400 flex-shrink-0"><Bot size={18}/></div>
                  <div className="p-3 rounded-xl bg-slate-800 text-slate-400 text-sm flex items-center gap-2 rounded-bl-none">
                     <Loader2 className="w-4 h-4 animate-spin" /> Analizuję dane...
                  </div>
               </div>
            )}
          </div>

          <div className="p-3 border-t border-slate-700 bg-slate-800 flex gap-2">
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Zapytaj asystenta..."
              className="flex-1 bg-slate-900 text-white rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500 border border-slate-700"
            />
            <button 
              onClick={sendMessage}
              disabled={isLoading || !input.trim()}
              className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white p-2 rounded-lg transition-colors flex-shrink-0"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}
    </>
  );
}
