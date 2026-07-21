'use client';

import React from 'react';
import useSWR from 'swr';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell
} from 'recharts';
import { Trees,  BarChart2, Award } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper7Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper7', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie wyników klasyfikacji (Random Forest vs SVM)...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 7.</div>;

  const { random_forest = {}, svm = {}, feature_importance = [] } = data;

  const featData = (Array.isArray(feature_importance) ? feature_importance : []).map(([feat, imp]: [string, number]) => ({
    feature: feat,
    importance: imp
  }));

  const rfAcc = random_forest.accuracy ?? 0;
  const svmAcc = svm.accuracy ?? 0;
  const rfCm = random_forest.confusion_matrix ?? [[0, 0], [0, 0]];
  const svmCm = svm.confusion_matrix ?? [[0, 0], [0, 0]];

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <Trees className="w-6 h-6 text-emerald-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">Klasyfikacja Kierunku Ceny (Random Forest vs SVM & Wagi Cech)</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Artykuł weryfikuje zdolność algorytmów uczenia maszynowego (<span className="text-emerald-400 font-semibold">Lasów Losowych</span> oraz <span className="text-blue-400 font-semibold">Maszyn Wektorów Nośnych</span>) do klasyfikacji binarnej kierunku zmiany ceny BTC na następny dzień (<span className="font-mono text-xs">dir_t1</span>). Kluczowym wnioskiem badania jest empiryczna ocena, które wskaźniki (zarówno techniczne, jak i sieciowe on-chain) posiadają największą moc dystrybucyjną i decyzyjną w lesie decyzyjnym.
            </p>
          </div>
        </div>
      </div>

      {/* 1. Kafelki porównawcze modeli */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Random Forest */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg relative overflow-hidden">
          <div className="flex justify-between items-start mb-4">
            <div>
              <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Random Forest (Las Losowy)</span>
              <div className="text-3xl font-bold font-mono text-white mt-1">{(rfAcc * 100).toFixed(1)}% <span className="text-xs font-normal text-slate-400">Dokładność (Accuracy)</span></div>
            </div>
            {rfAcc >= svmAcc && (
              <span className="px-2.5 py-1 bg-emerald-900/40 text-emerald-400 border border-emerald-800 rounded-full text-xs font-semibold flex items-center gap-1">
                <Award className="w-3.5 h-3.5" /> Lider
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-4 text-xs text-slate-300 border-t border-slate-800/80 pt-4">
            <div>Precision: <span className="font-mono font-bold text-white">{((random_forest.precision ?? 0) * 100).toFixed(1)}%</span></div>
            <div>Recall: <span className="font-mono font-bold text-white">{((random_forest.recall ?? 0) * 100).toFixed(1)}%</span></div>
          </div>
          {/* Confusion matrix mini */}
          <div className="mt-4 pt-3 border-t border-slate-800/50 text-xs">
            <div className="text-slate-400 mb-2">Macierz pomyłek (Confusion Matrix):</div>
            <div className="grid grid-cols-2 gap-2 text-center font-mono">
              <div className="bg-slate-950 p-2 rounded border border-slate-800 text-emerald-400 font-bold">TN: {rfCm[0][0]}</div>
              <div className="bg-slate-950 p-2 rounded border border-slate-800 text-red-400">FP: {rfCm[0][1]}</div>
              <div className="bg-slate-950 p-2 rounded border border-slate-800 text-red-400">FN: {rfCm[1][0]}</div>
              <div className="bg-slate-950 p-2 rounded border border-slate-800 text-emerald-400 font-bold">TP: {rfCm[1][1]}</div>
            </div>
          </div>
        </div>

        {/* SVM */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg relative overflow-hidden">
          <div className="flex justify-between items-start mb-4">
            <div>
              <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">Support Vector Machine (SVM)</span>
              <div className="text-3xl font-bold font-mono text-white mt-1">{(svmAcc * 100).toFixed(1)}% <span className="text-xs font-normal text-slate-400">Dokładność (Accuracy)</span></div>
            </div>
            {svmAcc > rfAcc && (
              <span className="px-2.5 py-1 bg-blue-900/40 text-blue-400 border border-blue-800 rounded-full text-xs font-semibold flex items-center gap-1">
                <Award className="w-3.5 h-3.5" /> Lider
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-4 text-xs text-slate-300 border-t border-slate-800/80 pt-4">
            <div>Precision: <span className="font-mono font-bold text-white">{((svm.precision ?? 0) * 100).toFixed(1)}%</span></div>
            <div>Recall: <span className="font-mono font-bold text-white">{((svm.recall ?? 0) * 100).toFixed(1)}%</span></div>
          </div>
          {/* Confusion matrix mini */}
          <div className="mt-4 pt-3 border-t border-slate-800/50 text-xs">
            <div className="text-slate-400 mb-2">Macierz pomyłek (Confusion Matrix):</div>
            <div className="grid grid-cols-2 gap-2 text-center font-mono">
              <div className="bg-slate-950 p-2 rounded border border-slate-800 text-blue-400 font-bold">TN: {svmCm[0][0]}</div>
              <div className="bg-slate-950 p-2 rounded border border-slate-800 text-red-400">FP: {svmCm[0][1]}</div>
              <div className="bg-slate-950 p-2 rounded border border-slate-800 text-red-400">FN: {svmCm[1][0]}</div>
              <div className="bg-slate-950 p-2 rounded border border-slate-800 text-blue-400 font-bold">TP: {svmCm[1][1]}</div>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Wykres ważności cech (Feature Importance) */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-emerald-400 mb-4 flex items-center gap-2">
          <BarChart2 className="w-5 h-5" />
          Ważność Cech w Algorytmie Lasu Losowego (Gini Feature Importance)
        </h3>
        <p className="text-slate-400 text-sm mb-6">Ranking zmiennych o największym wpływie na redukcję entropii / niedokładności klasyfikacji kierunku rynku.</p>
        <div className="h-[320px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={featData} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis type="number" stroke="#64748b" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <YAxis type="category" dataKey="feature" stroke="#cbd5e1" tick={{ fontSize: 13, fontFamily: 'monospace' }} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} formatter={(v: unknown) => [`${(Number(v) * 100).toFixed(2)}%`, 'Waga cechy']} />
              <Bar dataKey="importance" name="Waga cechy" radius={[0, 6, 6, 0]}>
                {featData.map((entry: unknown, idx: number) => (
                  <Cell key={`cell-${idx}`} fill={idx === 0 ? '#10b981' : idx === 1 ? '#34d399' : '#059669'} opacity={1 - idx * 0.08} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
