'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend
} from 'recharts';
import { BrainCircuit, Layers, Activity } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper5Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper5', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie prognoz czasowych (Facebook Prophet)...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 5.</div>;

  const { metrics = {}, predictions = [] } = data;

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <Layers className="w-6 h-6 text-indigo-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">Prognozowanie Szeregów Czasowych ze Zmiennymi Egzogenicznymi (Facebook Prophet)</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Model <span className="text-indigo-400 font-semibold">Prophet</span> oparty na dekompozycji addytywnej wzbogaconej o zewnętrzne regresory on-chain (m.in. wolumen transakcyjny oraz wskaźniki sieciowe). Algorytm wykorzystuje silnik <span className="font-mono text-xs text-amber-400">CmdStan</span> do estymacji parametrów sezonowości tygodniowej i rocznej oraz generuje 30-dniową prognozę <span className="italic">out-of-sample</span> weryfikowaną na rzeczywistych danych cenowych.
            </p>
          </div>
        </div>
      </div>

      {/* 1. Kafelki z błędami predykcji */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl relative overflow-hidden shadow-lg flex items-center justify-between">
          <div>
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Średni Błąd Bezwzględny (MAE)</div>
            <div className="text-3xl font-bold font-mono text-white mt-1">${metrics.MAE?.toFixed(2) ?? 'N/A'}</div>
            <p className="text-xs text-slate-500 mt-2">Średnia różnica między prognozą a faktyczną ceną USD w oknie testowym.</p>
          </div>
          <Activity className="w-12 h-12 text-indigo-500/20" />
        </div>

        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl relative overflow-hidden shadow-lg flex items-center justify-between">
          <div>
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Pierwiastek Błędu Średniokwadratowego (RMSE)</div>
            <div className="text-3xl font-bold font-mono text-amber-400 mt-1">${metrics.RMSE?.toFixed(2) ?? 'N/A'}</div>
            <p className="text-xs text-slate-500 mt-2">Wrażliwość na duże odchylenia (outliery) w predykcji trendu.</p>
          </div>
          <BrainCircuit className="w-12 h-12 text-amber-500/20" />
        </div>
      </div>

      {/* 2. Wykres zderzający prognozę vs wartość rzeczywistą */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-indigo-400 mb-4 flex items-center gap-2">
          <BrainCircuit className="w-5 h-5" />
          Prognoza Out-of-Sample (yhat vs Actual Price)
        </h3>
        <p className="text-slate-400 text-sm mb-6">Zestawienie rzeczywistego kursu Bitcoina ze ścieżką wyestymowaną przez model Prophet na zbiorze testowym.</p>
        <div className="h-[420px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={predictions} margin={{ top: 10, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('T')[0]} />
              <YAxis stroke="#94a3b8" domain={['auto', 'auto']} tickFormatter={(val) => `$${(val/1000).toFixed(1)}k`} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} formatter={(val: unknown) => [`$${Number(val).toFixed(2)}`, '']} />
              <Legend verticalAlign="top" height={36} wrapperStyle={{ paddingBottom: '10px' }} />
              <Line type="monotone" dataKey="actual" stroke="#38bdf8" strokeWidth={2.5} dot={false} name="Faktyczna Cena (Actual USD)" />
              <Line type="monotone" dataKey="predicted" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Prognoza Prophet (Predicted yhat)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
