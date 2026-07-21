'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, BarChart, Bar, Cell
} from 'recharts';
import { Cpu, Zap, Layers, Activity, Award } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper6Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper6', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie prognoz zmienności (Transformer vs LSTM)...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 6.</div>;

  const { sequence_length = 14, epochs_trained = 5, test_mse = {}, time_series = [] } = data;

  const mseData = [
    { model: 'Transformer (Multi-Head Attention)', mse: test_mse.Transformer ?? 0, color: '#38bdf8' },
    { model: 'LSTM (Recurrent Neural Network)', mse: test_mse.LSTM ?? 0, color: '#f43f5e' }
  ];

  const betterModel = (test_mse.Transformer ?? 0) <= (test_mse.LSTM ?? 999) ? 'Transformer' : 'LSTM';

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <Cpu className="w-6 h-6 text-purple-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">Prognozowanie Zmienności Zrealizowanej: Architektura Transformer vs LSTM</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Porównanie skuteczności głębokich sieci neuronowych w przewidywaniu zmienności zrealizowanej <span className="font-mono text-xs text-amber-400">rv_t</span>. Model oparciu o mechanizm uwag (Self-Attention / <span className="text-purple-400 font-semibold">Transformer</span>) analizuje sekwencje o długości <span className="font-mono text-xs">{sequence_length}</span> dni i konfrontowany jest z klasyczną architekturą rekurrencyjną <span className="text-red-400 font-semibold">LSTM</span> (Long Short-Term Memory).
            </p>
          </div>
        </div>
      </div>

      {/* 1. Kafelki i porównanie błędów MSE */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg flex flex-col justify-between">
          <div>
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Parametry Uczenia (Hyperparameters)</div>
            <div className="space-y-3 mt-4">
              <div className="flex justify-between items-center pb-2 border-b border-slate-800 text-sm">
                <span className="text-slate-400">Długość okna sekwencji (Sequence Len):</span>
                <span className="font-mono font-bold text-white">{sequence_length} dni</span>
              </div>
              <div className="flex justify-between items-center pb-2 border-b border-slate-800 text-sm">
                <span className="text-slate-400">Epoki treningowe (Epochs):</span>
                <span className="font-mono font-bold text-white">{epochs_trained}</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">Zwycięska architektura:</span>
                <span className="font-semibold text-emerald-400 flex items-center gap-1">
                  <Award className="w-4 h-4" /> {betterModel}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Wykres barowy MSE */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg lg:col-span-2">
          <h3 className="font-bold text-lg text-purple-400 mb-3 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Błąd Średniokwadratowy na Zbiorze Testowym (Test MSE - niższy = lepszy)
          </h3>
          <div className="h-[180px] w-full mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mseData} layout="vertical" margin={{ top: 5, right: 30, left: 160, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#64748b" tickFormatter={(v) => v.toExponential(2)} />
                <YAxis type="category" dataKey="model" stroke="#cbd5e1" tick={{ fontSize: 12, fill: '#e2e8f0' }} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} formatter={(v: any) => [Number(v).toExponential(4), 'MSE']} />
                <Bar dataKey="mse" radius={[0, 6, 6, 0]}>
                  {mseData.map((entry: any, idx: number) => (
                    <Cell key={`cell-${idx}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 2. Wykres zderzający prognozy Transformer i LSTM vs Actual */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-purple-400 mb-4 flex items-center gap-2">
          <Cpu className="w-5 h-5" />
          Szereg Czasowy Zmienności: Transformer vs LSTM vs Zmienność Faktyczna (Actual RV)
        </h3>
        <p className="text-slate-400 text-sm mb-6">Wizualizacja dopasowania modeli deep learningowych do nagłych skoków zmienności na rynku kapitałowym BTC.</p>
        <div className="h-[420px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={time_series} margin={{ top: 10, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('T')[0]} />
              <YAxis stroke="#94a3b8" tickFormatter={(val) => val.toFixed(3)} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} formatter={(val: any) => [Number(val).toFixed(5), '']} />
              <Legend verticalAlign="top" height={36} wrapperStyle={{ paddingBottom: '10px' }} />
              <Line type="monotone" dataKey="Actual_rv" stroke="#e2e8f0" strokeWidth={2.5} dot={false} name="Zmienność Rzeczywista (Actual RV)" />
              <Line type="monotone" dataKey="Pred_Transformer" stroke="#38bdf8" strokeWidth={2} dot={false} name="Prognoza Transformer" />
              <Line type="monotone" dataKey="Pred_LSTM" stroke="#f43f5e" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Prognoza LSTM" opacity={0.8} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
